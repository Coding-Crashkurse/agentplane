"""Endpoint serving (SPEC §6.5): A2A agents and MCP servers behind the gateway.

The runtime mounts one ASGI sub-app per deployed flow under
``/a2a/{name}`` (a2a-sdk server, A2A v1.0 wire format) or ``/mcp/{name}``
(FastMCP, streamable HTTP). Ephemeral draft endpoints live under
``/a2a/_draft/{name}`` with a TTL and never touch the registry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, MutableMapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.owner_resolver import resolve_user_scope
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import DatabaseTaskStore, InMemoryTaskStore, TaskStore, TaskUpdater
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
from a2a.types.a2a_pb2 import ListTasksRequest, Role
from fastapi import HTTPException
from fastmcp import FastMCP
from fastmcp.tools import Tool, ToolResult
from opentelemetry import trace
from pydantic import PrivateAttr
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.applications import Starlette
from starlette.authentication import SimpleUser
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from agentplane_core import (
    FlowDefinition,
    JsonObject,
    LlmCallNode,
    StartNode,
    single_required_string_input,
)
from agentplane_runtime.auth import AccessScope, Authenticator
from agentplane_runtime.directory import AgentDirectory
from agentplane_runtime.engine import (
    CALL_DEPTH_METADATA_KEY,
    ConversationTurn,
    ExecutionContext,
    FlowRunner,
    _as_text,
    caller_token_var,
)
from agentplane_runtime.resources import ResourceService
from agentplane_runtime.settings import EPHEMERAL_TTL_S, RuntimeSettings

logger = logging.getLogger(__name__)

AGENT_CARD_PATH = "/.well-known/agent-card.json"

# Caller display name for trace attribution, set by the EndpointGuard and read
# by the executor: the HTTP spans are dropped as noise by the collector, so
# user attribution must live on the flow span itself.
_caller_display: ContextVar[str] = ContextVar("agentplane_caller_display", default="")

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


def _wants_history(defn: FlowDefinition) -> bool:
    return any(isinstance(node, LlmCallNode) and node.config.history for node in defn.nodes)


async def _load_conversation(
    store: TaskStore,
    context_id: str,
    current_task_id: str,
    call_context: ServerCallContext,
) -> list[ConversationTurn]:
    """Prior turns of the conversation, oldest first (SPEC §6.5).

    User text comes from each task's history, the reply from its artifacts —
    the same reconstruction the chat UI uses. The current task (already saved
    with the incoming message) is excluded.
    """
    request = ListTasksRequest(
        context_id=context_id, page_size=100, history_length=200, include_artifacts=True
    )
    response = await store.list(request, call_context)
    turns: list[ConversationTurn] = []
    for task in reversed(list(response.tasks)):  # store lists newest first
        if task.id == current_task_id:
            continue
        user_text = "".join(
            part.text
            for message in task.history
            if message.role == Role.ROLE_USER
            for part in message.parts
            if part.text
        )
        reply_text = "".join(
            part.text for artifact in task.artifacts for part in artifact.parts if part.text
        )
        if user_text:
            turns.append(("user", user_text))
        if reply_text:
            turns.append(("assistant", reply_text))
    return turns


class FlowAgentExecutor(AgentExecutor):
    """Runs the flow for each A2A request; streams tokens for stream:true nodes."""

    def __init__(
        self,
        runner_factory: Callable[[], FlowRunner],
        defn: FlowDefinition,
        task_store: TaskStore | None = None,
        max_call_depth: int = 5,
    ) -> None:
        self._runner_factory = runner_factory
        self._defn = defn
        self._task_store = task_store
        self._max_call_depth = max_call_depth

    def _incoming_call_depth(self, context: RequestContext) -> int:
        """Orchestration depth carried in the A2A message metadata (0 = direct)."""
        if context.message is None:
            return 0
        fields = context.message.metadata.fields
        if CALL_DEPTH_METADATA_KEY not in fields:
            return 0
        return int(fields[CALL_DEPTH_METADATA_KEY].number_value)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        if context.current_task is None:
            # Async workflow: the Task object must be enqueued before updates.
            # The incoming message goes into the task history so persisted
            # tasks can restore the full conversation (SPEC §6.5).
            await event_queue.enqueue_event(
                Task(
                    id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                    history=[context.message] if context.message is not None else None,
                )
            )
        # The A2A context is the conversation: expose it as the OTel session id
        # so tracing UIs group a user's exchanges (SPEC §12).
        if context_id:
            trace.get_current_span().set_attribute("session.id", context_id)
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()

        # Recursion guard for orchestrator chains (A calls B calls A ...):
        # every runtime enforces the limit on receipt, so a loop dies at the
        # perimeter no matter which runtime started it.
        call_depth = self._incoming_call_depth(context)
        if call_depth > self._max_call_depth:
            await updater.failed(
                updater.new_agent_message(
                    [Part(text=f"agent call depth limit ({self._max_call_depth}) exceeded")]
                )
            )
            return

        async def stream_chunk(delta: str) -> None:
            await updater.update_status(
                TaskState.TASK_STATE_WORKING,
                message=updater.new_agent_message([Part(text=delta)]),
            )

        conversation: list[ConversationTurn] | None = None
        if self._task_store is not None and context_id and _wants_history(self._defn):
            try:
                conversation = await _load_conversation(
                    self._task_store, context_id, task_id, context.call_context
                )
            except Exception:
                # History is best-effort: a store hiccup must not fail the chat.
                logger.warning("conversation history unavailable", exc_info=True)

        # The flow span is the trace's anchor (the HTTP spans around it are
        # dropped as noise), so it carries the caller and the conversation.
        trace_attributes: dict[str, str] = {}
        if context_id:
            trace_attributes["session.id"] = context_id
        if caller := _caller_display.get():
            trace_attributes["user.id"] = caller

        runner = self._runner_factory()
        try:
            inputs = bind_message_to_inputs(self._defn, context.get_user_input())
            result = await runner.execute(
                inputs,
                stream=stream_chunk,
                trace_attributes=trace_attributes,
                conversation=conversation,
                call_depth=call_depth,
            )
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
    task_store: TaskStore | None = None,
    max_call_depth: int = 5,
) -> Starlette:
    """One Starlette app per A2A-exposed flow: card + JSON-RPC routes.

    ``task_store`` enables persistent conversations (SPEC §6.5): with a
    database-backed store, tasks survive restarts and clients can restore
    chat history via the A2A ``ListTasks``/``GetTask`` methods. Defaults to
    in-memory (today's behavior).
    """
    card = build_agent_card(defn, public_url, version)
    # The executor shares the store so LLM nodes with `history: true` can load
    # the conversation's prior turns (works for the in-memory store too).
    store = task_store or InMemoryTaskStore()
    handler = DefaultRequestHandler(
        agent_executor=FlowAgentExecutor(
            runner_factory, defn, task_store=store, max_call_depth=max_call_depth
        ),
        task_store=store,
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


class EndpointGuard:
    """Per-flow invocation authorization (SPEC §7.1).

    With ``auth_mode=oidc`` every request to a served endpoint must carry a JWT
    whose subject is the flow's owner, a member of its group, or an admin —
    "who may call" follows the same predicate as "who may edit". Discovery
    stays public (``public_paths``, GET/HEAD only): agent cards carry no
    secrets and the registry health job fetches them unauthenticated. With
    auth off this is a transparent pass-through.
    """

    def __init__(
        self,
        app: ASGIApp,
        authenticator: Authenticator,
        owner: str,
        group: str,
        public_paths: frozenset[str] = frozenset(),
    ) -> None:
        self._app = app
        self._authenticator = authenticator
        self._owner = owner
        self._group = group
        self._public_paths = public_paths

    def _is_public(self, scope: Scope) -> bool:
        if scope.get("method") not in ("GET", "HEAD"):
            return False
        path: str = scope.get("path", "")
        root_path: str = scope.get("root_path", "")
        sub_path = path[len(root_path) :] if path.startswith(root_path) else path
        return (sub_path or "/") in self._public_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self._authenticator.mode != "oidc" or self._is_public(scope):
            await self._app(scope, receive, send)
            return
        request = Request(scope)
        try:
            principal = await self._authenticator.authenticate(request)
        except HTTPException as exc:
            response: PlainTextResponse | JSONResponse = JSONResponse(
                {"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers
            )
            await response(scope, receive, send)
            return
        if not AccessScope.for_caller(principal, "oidc").allows(self._owner, self._group):
            response = JSONResponse(
                {"detail": "not authorized to call this endpoint"},
                status_code=403,
            )
            await response(scope, receive, send)
            return
        # Attribute the trace to the caller (SPEC §12): `user.id` is the OTel
        # convention Langfuse maps to the trace's user. The username reads
        # better in tracing UIs; the subject is the fallback identity.
        trace.get_current_span().set_attribute("user.id", principal.username or principal.sub)
        _caller_display.set(principal.username or principal.sub)
        # Keep the raw token for delegation: orchestrator sub-agent calls act
        # on behalf of this caller (engine.caller_token_var), so the callee's
        # own EndpointGuard authorizes the end user, not the runtime.
        _, _, bearer = request.headers.get("Authorization", "").partition(" ")
        caller_token_var.set(bearer)
        # Expose the caller to the a2a app (SPEC §6.5): the task store scopes
        # persisted conversations by this user (StarletteUser reads
        # display_name).
        scope["user"] = SimpleUser(principal.sub)
        await self._app(scope, receive, send)


# Discovery surface that stays public on A2A endpoints (GET/HEAD only).
_A2A_PUBLIC_PATHS = frozenset({"/", "/.well-known/agent-card.json"})


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

    def __init__(
        self,
        resources: ResourceService,
        settings: RuntimeSettings,
        authenticator: Authenticator | None = None,
        engine: AsyncEngine | None = None,
        directory: AgentDirectory | None = None,
    ) -> None:
        self._resources = resources
        self._settings = settings
        self._authenticator = authenticator or Authenticator(settings)
        self._engine = engine
        self._directory = directory
        self.a2a = PathDispatcher()
        self.mcp = PathDispatcher()
        self._endpoints: dict[str, Endpoint] = {}

    def _persistent_task_store(self, flow_key: str) -> TaskStore | None:
        """A database task store for one endpoint (``TASK_STORE=database``).

        All endpoints share the sdk-owned ``tasks`` table (created on first
        use, deliberately outside the Alembic chain — library schema). The
        owner column scopes rows to ``{flow}::{caller}``, so ``ListTasks`` /
        ``GetTask`` never leak conversations across endpoints or users. With
        auth off the caller part is empty — one shared history per flow.
        """
        if self._settings.task_store != "database" or self._engine is None:
            return None

        def owner(context: ServerCallContext) -> str:
            return f"{flow_key}::{resolve_user_scope(context)}"

        return DatabaseTaskStore(self._engine, owner_resolver=owner)

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
                            agents=self._directory,
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
        owner: str = "",
        group: str = "",
    ) -> Endpoint:
        """(Re)start the endpoint for a definition at a given version.

        ``owner``/``group`` scope who may invoke the endpoint when auth is on
        (same predicate as editing); with auth off the guard passes through.
        """
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
            self.mcp.mount(key, EndpointGuard(http_app, self._authenticator, owner, group))
        else:
            # Drafts stay in-memory: ephemeral endpoints must not persist tasks.
            app = build_a2a_app(
                defn,
                public_url,
                runner_factory,
                version_label or str(version),
                task_store=None if ephemeral else self._persistent_task_store(defn.name),
                max_call_depth=self._settings.max_agent_call_depth,
            )
            self.a2a.mount(
                key,
                EndpointGuard(
                    app, self._authenticator, owner, group, public_paths=_A2A_PUBLIC_PATHS
                ),
            )

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
    "EndpointGuard",
    "EndpointManager",
    "FlowAgentExecutor",
    "PathDispatcher",
    "bind_message_to_inputs",
    "build_a2a_app",
    "build_agent_card",
    "build_mcp_server",
]
