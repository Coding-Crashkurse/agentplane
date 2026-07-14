"""Endpoint serving (SPEC §6.5): A2A agents and MCP servers behind the gateway.

The runtime mounts one ASGI sub-app per deployed flow under
``/a2a/{name}`` (a2a-sdk server, A2A v1.0 wire format) or ``/mcp/{name}``
(FastMCP, streamable HTTP). Ephemeral draft endpoints live under
``/a2a/_draft/{name}`` with a TTL and never touch the registry.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Part,
    Task,
    TaskState,
    TaskStatus,
)
from fastmcp import FastMCP
from fastmcp.tools import Tool, ToolResult
from pydantic import PrivateAttr
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from agentplane_core import (
    FlowDefinition,
    JsonObject,
    StartNode,
    single_required_string_input,
)
from agentplane_runtime.engine import ExecutionContext, FlowRunner, _as_text
from agentplane_runtime.resources import ResourceService
from agentplane_runtime.settings import EPHEMERAL_TTL_S, RuntimeSettings

AGENT_CARD_PATH = "/.well-known/agent-card.json"

# ASGI protocol types — inherently loose (the ASGI spec is dict-based).
Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def build_agent_card(defn: FlowDefinition, public_url: str, version: str = "1") -> AgentCard:
    """Derive the A2A card from the definition (SPEC §6.5).

    ``version`` is the published version label when the publisher chose one,
    else the deploy counter — the card always carries the version that is
    actually being served.
    """
    skills = [
        AgentSkill(
            id=defn.name,
            name=defn.display_name or defn.name,
            description=defn.description or defn.display_name or defn.name,
            tags=list(defn.tags) or ["flow"],
            examples=list(defn.expose.examples),
        )
    ]
    return AgentCard(
        name=defn.name,
        description=defn.description,
        version=version,
        supported_interfaces=[
            AgentInterface(url=public_url, protocol_binding="JSONRPC", protocol_version="1.0")
        ],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain"],
        skills=skills,
    )


def bind_message_to_inputs(defn: FlowDefinition, text: str) -> JsonObject:
    """A2A input binding (SPEC §6.4).

    Message text binds to the single required string property when there is
    exactly one; otherwise the text must be a JSON object matching the schema.
    """
    single = single_required_string_input(defn)
    if single is not None:
        return {single: text}
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        raise ValueError(
            "flow input schema has multiple properties; send a JSON object message"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("message must be a JSON object matching the flow input schema")
    return parsed


class FlowAgentExecutor(AgentExecutor):
    """Runs the flow for each A2A request; streams tokens for stream:true nodes."""

    def __init__(self, runner_factory: Callable[[], FlowRunner], defn: FlowDefinition) -> None:
        self._runner_factory = runner_factory
        self._defn = defn

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        if context.current_task is None:
            # Async workflow: the Task object must be enqueued before updates.
            await event_queue.enqueue_event(
                Task(
                    id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                )
            )
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()

        async def stream_chunk(delta: str) -> None:
            await updater.update_status(
                TaskState.TASK_STATE_WORKING,
                message=updater.new_agent_message([Part(text=delta)]),
            )

        runner = self._runner_factory()
        try:
            inputs = bind_message_to_inputs(self._defn, context.get_user_input())
            result = await runner.execute(inputs, stream=stream_chunk)
        except ValueError as exc:
            await updater.failed(updater.new_agent_message([Part(text=str(exc))]))
            return
        except Exception as exc:
            await updater.failed(
                updater.new_agent_message([Part(text=f"flow execution failed: {exc}")])
            )
            return
        text = _as_text(result) if result is not None else ""
        await updater.add_artifact([Part(text=text)], name="output")
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id or "", context.context_id or "")
        await updater.cancel()


def build_a2a_app(
    defn: FlowDefinition,
    public_url: str,
    runner_factory: Callable[[], FlowRunner],
    version: str = "1",
) -> Starlette:
    """One Starlette app per A2A-exposed flow: card + JSON-RPC routes."""
    card = build_agent_card(defn, public_url, version)
    handler = DefaultRequestHandler(
        agent_executor=FlowAgentExecutor(runner_factory, defn),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    async def endpoint_info(_: Request) -> JSONResponse:
        """A browser GET on the endpoint lands here — the JSON-RPC binding is POST-only."""
        return JSONResponse(
            {
                "name": defn.name,
                "description": defn.description,
                "protocol": "A2A",
                "protocol_version": "1.0",
                "agent_card_url": f"{public_url}{AGENT_CARD_PATH}",
                "jsonrpc_url": public_url,
                "hint": (
                    "POST JSON-RPC 2.0 requests here with the header 'A2A-Version: 1.0'; "
                    "GET the agent card at agent_card_url."
                ),
            }
        )

    routes: list[Route] = [
        Route("/", endpoint_info, methods=["GET"]),
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, "/"),
    ]
    return Starlette(routes=routes)


class FlowTool(Tool):
    """FastMCP tool whose parameters come from ``start.input_schema``."""

    _runner_factory: Callable[[], FlowRunner] = PrivateAttr()

    def bind(self, runner_factory: Callable[[], FlowRunner]) -> FlowTool:
        self._runner_factory = runner_factory
        return self

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        runner = self._runner_factory()
        result = await runner.execute(dict(arguments))
        return ToolResult(content=_as_text(result) if result is not None else "")


def build_mcp_server(defn: FlowDefinition, runner_factory: Callable[[], FlowRunner]) -> FastMCP:
    """One FastMCP server per MCP-exposed flow, one tool per flow (SPEC §6.5)."""
    start = next(n for n in defn.nodes if isinstance(n, StartNode))
    tool_name = defn.expose.tool_name or defn.name.replace("-", "_")
    server: FastMCP = FastMCP(name=defn.name, instructions=defn.description)
    tool = FlowTool(
        name=tool_name,
        description=defn.expose.tool_description or defn.description,
        parameters=start.config.input_schema,
    ).bind(runner_factory)
    server.add_tool(tool)
    return server


class PathDispatcher:
    """Dispatches ``/{name}/...`` to per-flow ASGI apps; supports live add/remove."""

    def __init__(self) -> None:
        self._apps: dict[str, ASGIApp] = {}

    def mount(self, name: str, app: ASGIApp) -> None:
        self._apps[name] = app

    def unmount(self, name: str) -> None:
        self._apps.pop(name, None)

    def mounted(self) -> list[str]:
        return sorted(self._apps)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            return
        # Starlette convention: `path` is the full request path and `root_path`
        # the already-consumed prefix; route matching happens on the difference.
        path: str = scope.get("path", "")
        root_path: str = scope.get("root_path", "")
        route_path = path[len(root_path) :] if path.startswith(root_path) else path
        segments = [s for s in route_path.split("/") if s]
        name = ""
        consumed = 0
        if len(segments) > 1 and segments[0] == "_draft":
            name = f"_draft/{segments[1]}"
            consumed = 2
        elif segments:
            name = segments[0]
            consumed = 1
        app = self._apps.get(name)
        if app is None:
            response = PlainTextResponse("no such endpoint", status_code=404)
            await response(scope, receive, send)
            return
        child_scope = dict(scope)
        child_scope["root_path"] = root_path + "/" + "/".join(segments[:consumed])
        await app(child_scope, receive, send)


class LifespanHost:
    """Runs an ASGI app's lifespan in a dedicated task.

    anyio cancel scopes must be entered and exited in the same task, so the
    FastMCP http_app lifespan cannot live in an exit stack that is closed
    from elsewhere.
    """

    def __init__(self, app: Starlette) -> None:
        self._app = app
        self._started: asyncio.Event = asyncio.Event()
        self._stop: asyncio.Event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        async with self._app.router.lifespan_context(self._app):
            self._started.set()
            await self._stop.wait()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="endpoint-lifespan")
        await self._started.wait()

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        await self._task
        self._task = None


@dataclass
class Endpoint:
    name: str
    kind: str  # "a2a" | "mcp"
    version: int
    public_url: str
    version_label: str | None = None
    lifespan: LifespanHost | None = None
    expires_at: float | None = None  # ephemeral only


class EndpointManager:
    """Owns the /a2a and /mcp dispatchers and the running endpoints."""

    def __init__(self, resources: ResourceService, settings: RuntimeSettings) -> None:
        self._resources = resources
        self._settings = settings
        self.a2a = PathDispatcher()
        self.mcp = PathDispatcher()
        self._endpoints: dict[str, Endpoint] = {}

    def endpoint_for(self, name: str) -> Endpoint | None:
        return self._endpoints.get(name)

    def _runner_factory(self, defn: FlowDefinition, version: int) -> Callable[[], FlowRunner]:
        compiled: list[FlowRunner] = []

        def factory() -> FlowRunner:
            # compile once, reuse the graph; context is shared per endpoint
            if not compiled:
                compiled.append(
                    FlowRunner(
                        defn,
                        ExecutionContext(
                            resources=self._resources,
                            settings=self._settings,
                            flow_name=defn.name,
                            flow_version=version,
                        ),
                    )
                )
            return compiled[0]

        return factory

    def public_url(self, defn: FlowDefinition, *, ephemeral: bool = False) -> str:
        base = self._settings.public_base_url.rstrip("/")
        if ephemeral:
            return f"{base}/a2a/_draft/{defn.name}"
        return f"{base}/{defn.expose.kind}/{defn.name}"

    async def start(
        self,
        defn: FlowDefinition,
        version: int,
        *,
        version_label: str | None = None,
        ephemeral: bool = False,
    ) -> Endpoint:
        """(Re)start the endpoint for a definition at a given version."""
        key = f"_draft/{defn.name}" if ephemeral else defn.name
        await self.stop(key)
        public_url = self.public_url(defn, ephemeral=ephemeral)
        runner_factory = self._runner_factory(defn, version)
        lifespan: LifespanHost | None = None

        if defn.expose.kind == "mcp" and not ephemeral:
            server = build_mcp_server(defn, runner_factory)
            http_app = server.http_app(path="/", stateless_http=True)
            lifespan = LifespanHost(http_app)
            await lifespan.start()
            self.mcp.mount(key, http_app)
        else:
            app = build_a2a_app(defn, public_url, runner_factory, version_label or str(version))
            self.a2a.mount(key, app)

        endpoint = Endpoint(
            name=key,
            kind=defn.expose.kind if not ephemeral else "a2a",
            version=version,
            public_url=public_url,
            version_label=version_label,
            lifespan=lifespan,
            expires_at=time.monotonic() + EPHEMERAL_TTL_S if ephemeral else None,
        )
        self._endpoints[key] = endpoint
        return endpoint

    async def stop(self, name: str) -> None:
        endpoint = self._endpoints.pop(name, None)
        if endpoint is None:
            return
        if endpoint.kind == "mcp":
            self.mcp.unmount(name)
        else:
            self.a2a.unmount(name)
        if endpoint.lifespan is not None:
            await endpoint.lifespan.stop()

    async def stop_all(self) -> None:
        for name in list(self._endpoints):
            await self.stop(name)

    async def reap_expired(self) -> None:
        """Drop ephemeral endpoints past their TTL."""
        now = time.monotonic()
        for name, endpoint in list(self._endpoints.items()):
            if endpoint.expires_at is not None and endpoint.expires_at < now:
                await self.stop(name)


__all__ = [
    "AGENT_CARD_PATH",
    "EndpointManager",
    "FlowAgentExecutor",
    "PathDispatcher",
    "bind_message_to_inputs",
    "build_a2a_app",
    "build_agent_card",
    "build_mcp_server",
]
