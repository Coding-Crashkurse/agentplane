"""Flow execution engine (SPEC §6.4): a deployed version compiles to a LangGraph
graph once (cached); every node execution is one OTel span.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from operator import or_
from typing import Annotated, Protocol, TypedDict, TypeGuard, cast
from urllib.parse import urlsplit

import httpx
from langgraph.graph import END, START, StateGraph
from opentelemetry import propagate, trace
from pydantic import JsonValue

from agentplane_core import (
    AgentNode,
    AgentNodeConfig,
    Document,
    EndNode,
    FlowDefinition,
    JsonObject,
    LlmCallNode,
    McpServerResource,
    McpToolNode,
    ModelProviderResource,
    Node,
    RerankNode,
    RetrievalNode,
    RouterNode,
    RouterNodeConfig,
    RouterRule,
    StartNode,
    TemplateNode,
    VectorDBResource,
    input_ports,
    render_documents,
    split_port_ref,
)
from agentplane_runtime.directory import AgentDirectory, ResolvedAgent
from agentplane_runtime.llm import OpenAICompatibleClient
from agentplane_runtime.resources import ResourceService
from agentplane_runtime.settings import RuntimeSettings
from agentplane_runtime.vector import reader_for

tracer = trace.get_tracer("agentplane-runtime")

# A port value: JSON-serializable data or a list of retrieved documents.
PortValue = JsonValue | list[Document]

StreamCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class _ToolTarget:
    """Where an agent tool call routes: an MCP server url + auth + tool name."""

    url: str
    auth: str | None
    tool: str


class FlowError(RuntimeError):
    """A node failed during execution."""


# Per-run token stream callback. A ContextVar (not runner state) because the
# compiled runner is shared across concurrent requests.
_stream_var: ContextVar[StreamCallback | None] = ContextVar("agentplane_stream", default=None)

# One prior exchange of the caller's conversation: ("user"|"assistant", text).
ConversationTurn = tuple[str, str]

# Per-run conversation history (prior turns of the A2A context). Same
# ContextVar reasoning as the stream callback above.
_conversation_var: ContextVar[tuple[ConversationTurn, ...]] = ContextVar(
    "agentplane_conversation", default=()
)

# A2A message metadata key carrying the orchestration depth of a call chain.
CALL_DEPTH_METADATA_KEY = "agentplane_call_depth"

# Per-run orchestration depth of the *incoming* request (0 = called directly).
_call_depth_var: ContextVar[int] = ContextVar("agentplane_call_depth", default=0)

# The caller's raw bearer token, set by the serving layer's EndpointGuard.
# Sub-agent calls forward it, so the orchestrator can only reach agents the
# end user may call — the owner/group predicate applies transitively and no
# privilege is amplified.
caller_token_var: ContextVar[str] = ContextVar("agentplane_caller_token", default="")


class FlowState(TypedDict):
    """LangGraph state: values keyed by 'node_id.port', plus the nodes that ran."""

    values: Annotated[dict[str, PortValue], or_]
    executed: Annotated[set[str], or_]


class _NodeFn(Protocol):
    """A LangGraph node callable (named ``state`` parameter, async update)."""

    def __call__(self, state: FlowState) -> Awaitable[dict[str, object]]: ...


@dataclass
class ExecutionContext:
    """Per-run context: resource access and optional token streaming."""

    resources: ResourceService
    settings: RuntimeSettings
    stream: StreamCallback | None = None
    flow_name: str = ""
    flow_version: int = 0
    extra_attributes: dict[str, str] = field(default_factory=dict)
    agents: AgentDirectory | None = None


def _as_text(value: PortValue) -> str:
    """Implicit conversion to a text port (documents render, json stringifies)."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and all(isinstance(v, Document) for v in value):
        return render_documents([v for v in value if isinstance(v, Document)])
    if isinstance(value, list) and not value:
        return ""
    return json.dumps(value, ensure_ascii=False, default=str)


def _as_json(value: PortValue) -> JsonValue:
    if isinstance(value, list) and value and all(isinstance(v, Document) for v in value):
        return [v.model_dump(mode="json") for v in value if isinstance(v, Document)]
    if isinstance(value, list):
        return [
            item.model_dump(mode="json") if isinstance(item, Document) else item for item in value
        ]
    return value


class FlowRunner:
    """Compiles a FlowDefinition into a LangGraph graph and runs it."""

    def __init__(self, defn: FlowDefinition, context: ExecutionContext) -> None:
        self._defn = defn
        self._context = context
        self._nodes = defn.node_map()
        # inbound edges per node: input port -> source refs. A port may be fed
        # by several edges (branch convergence on `end.input`); at runtime the
        # first ref present in the state wins — with a router upstream exactly
        # one branch produced a value.
        self._inputs: dict[str, dict[str, list[str]]] = {node_id: {} for node_id in self._nodes}
        for edge in defn.edges:
            dst_node, dst_port = split_port_ref(edge.to)
            if dst_node in self._inputs:
                self._inputs[dst_node].setdefault(dst_port, []).append(edge.from_)
        self._graph = self._compile()

    def _compile(self) -> object:
        builder = StateGraph(FlowState)
        for node in self._defn.nodes:
            builder.add_node(node.id, self._executor_for(node))
        start = next(n for n in self._defn.nodes if isinstance(n, StartNode))
        end = next(n for n in self._defn.nodes if isinstance(n, EndNode))
        routers = [n for n in self._defn.nodes if isinstance(n, RouterNode)]
        router_ids = {n.id for n in routers}
        builder.add_edge(START, start.id)
        seen: set[tuple[str, str]] = set()
        for edge in self._defn.edges:
            src, _ = split_port_ref(edge.from_)
            dst, _ = split_port_ref(edge.to)
            if src in router_ids:
                continue  # routers wire conditionally below
            if (src, dst) not in seen and src in self._nodes and dst in self._nodes:
                builder.add_edge(src, dst)
                seen.add((src, dst))
        for router in routers:
            self._wire_router(builder, router)
        # output_from creates an implicit dependency when no edge exists yet
        out_src, _ = split_port_ref(end.config.output_from)
        if (out_src, end.id) not in seen and out_src in self._nodes and out_src != end.id:
            builder.add_edge(out_src, end.id)
        builder.add_edge(end.id, END)
        return builder.compile()

    def _wire_router(self, builder: StateGraph[FlowState], router: RouterNode) -> None:
        """Only the chosen branch's downstream nodes execute (conditional edges)."""
        targets_by_branch: dict[str, list[str]] = {}
        for edge in self._defn.edges:
            src, src_port = split_port_ref(edge.from_)
            if src != router.id:
                continue
            dst, _ = split_port_ref(edge.to)
            if dst in self._nodes:
                targets_by_branch.setdefault(src_port, []).append(dst)
        router_id = router.id

        def selector(state: FlowState) -> list[str] | str:
            values = state["values"]
            for branch, targets in targets_by_branch.items():
                if f"{router_id}.{branch}" in values:
                    return targets
            return END  # chosen branch has no wired targets — this path ends

        path_map = sorted({t for targets in targets_by_branch.values() for t in targets})
        builder.add_conditional_edges(router_id, selector, [*path_map, END])

    def _executor_for(self, node: Node) -> _NodeFn:
        async def run_node(state: FlowState) -> dict[str, object]:
            if node.id in state["executed"] or not self._ready(node, state["values"]):
                return {}
            with tracer.start_as_current_span(
                "agentplane.node",
                attributes={
                    "flow.name": self._context.flow_name,
                    "flow.version": self._context.flow_version,
                    "node.id": node.id,
                    "node.type": node.type,
                },
            ):
                outputs = await self._run(node, state["values"])
            return {"values": outputs, "executed": {node.id}}

        return run_node

    def _ready(self, node: Node, values: dict[str, PortValue]) -> bool:
        """Whether every wired input port of ``node`` carries a value yet.

        LangGraph (Pregel) triggers a node as soon as *any* predecessor wrote, so
        a node fed from different graph depths would run once per predecessor —
        the first time with inputs still missing, i.e. a wasted LLM call. Waiting
        until all wired ports are filled turns such a fan-in into a join, and it
        gates router branches for free: the port fed by a branch that was not
        chosen never fills, so the node stays dormant. Every predecessor write
        re-triggers the node, so a skipped run is retried once the value arrives.
        """
        return all(
            any(source_ref in values for source_ref in source_refs)
            for source_refs in self._inputs.get(node.id, {}).values()
        )

    def _gather_inputs(self, node: Node, values: dict[str, PortValue]) -> dict[str, PortValue]:
        gathered: dict[str, PortValue] = {}
        for port, source_refs in self._inputs.get(node.id, {}).items():
            for source_ref in source_refs:
                if source_ref in values:
                    gathered[port] = values[source_ref]
                    break
        return gathered

    async def _run(  # noqa: PLR0911 - one return per node type
        self, node: Node, values: dict[str, PortValue]
    ) -> dict[str, PortValue]:
        inputs = self._gather_inputs(node, values)
        match node:
            case StartNode():
                return {}  # start outputs are seeded before invocation
            case EndNode():
                if node.config.output_from:
                    return {f"{node.id}.output": values.get(node.config.output_from)}
                # empty output_from: take the value wired into `input` —
                # branched flows feed end from whichever branch executed
                return {f"{node.id}.output": inputs.get("input")}
            case LlmCallNode():
                return await self._run_llm(node, inputs)
            case AgentNode():
                return await self._run_agent(node, inputs)
            case RetrievalNode():
                return await self._run_retrieval(node, inputs)
            case RerankNode():
                return await self._run_rerank(node, inputs)
            case McpToolNode():
                return await self._run_mcp_tool(node, inputs)
            case RouterNode():
                value = inputs.get("input")
                branch = _route(node.config, value)
                return {f"{node.id}.{branch}": value}
            case TemplateNode():
                rendered = {port: _as_text(value) for port, value in inputs.items()}
                for port in input_ports(node):
                    rendered.setdefault(port, "")
                return {f"{node.id}.text": _format_prompt(node.config.text, rendered)}

    async def _llm_client(self, resource_name: str) -> tuple[OpenAICompatibleClient, str]:
        resource = await self._context.resources.get_raw(resource_name)
        if not isinstance(resource, ModelProviderResource):
            raise FlowError(f"resource {resource_name!r} is not a model provider")
        api_key = ""
        if resource.api_key_secret:
            try:
                api_key = await self._context.resources.secret_value(
                    resource_name, "api_key_secret"
                )
            except KeyError:
                api_key = ""
        base_url = resource.base_url or self._context.settings.llm_base_url
        if not base_url:
            raise FlowError(
                f"model provider {resource_name!r} has no base_url and no "
                "AGENTPLANE_RUNTIME_LLM_BASE_URL default is configured"
            )
        client = OpenAICompatibleClient(
            base_url, api_key, timeout=self._context.settings.http_timeout_s
        )
        return client, resource.default_model

    async def _run_llm(
        self, node: LlmCallNode, inputs: dict[str, PortValue]
    ) -> dict[str, PortValue]:
        config = node.config
        rendered = {port: _as_text(value) for port, value in inputs.items()}
        for port in input_ports(node):
            rendered.setdefault(port, "")
        prompt = _format_prompt(config.prompt, rendered)
        system_prompt = _format_prompt(config.system_prompt, rendered)
        client, default_model = await self._llm_client(config.resource)
        model = config.model or default_model
        if not model:
            raise FlowError(f"node {node.id!r}: no model configured (node or resource default)")

        turns: tuple[ConversationTurn, ...] = ()
        if config.history:
            # `history_max_turns` counts exchanges (user + assistant pairs).
            turns = _conversation_var.get()[-2 * config.history_max_turns :]

        stream_cb = _stream_var.get() or self._context.stream
        if config.stream and stream_cb is not None:
            chunks: list[str] = []
            async for delta in client.stream(
                model, prompt, system_prompt, config.structured_output, turns=turns
            ):
                chunks.append(delta)
                await stream_cb(delta)
            text = "".join(chunks)
        else:
            text = await client.complete(
                model, prompt, system_prompt, config.structured_output, turns=turns
            )

        outputs: dict[str, PortValue] = {f"{node.id}.text": text}
        if config.structured_output is not None:
            try:
                outputs[f"{node.id}.json"] = json.loads(text)
            except ValueError as exc:
                raise FlowError(f"node {node.id!r}: structured output is not JSON") from exc
        return outputs

    async def _run_retrieval(
        self, node: RetrievalNode, inputs: dict[str, PortValue]
    ) -> dict[str, PortValue]:
        config = node.config
        query = _as_text(inputs.get("query", ""))
        resource = await self._context.resources.get_raw(config.resource)
        if not isinstance(resource, VectorDBResource):
            raise FlowError(f"resource {config.resource!r} is not a vector DB")
        embed_client, _ = await self._llm_client(resource.embedding.resource)
        vector = await embed_client.embed(resource.embedding.model, query)
        api_key, dsn = "", ""
        if resource.api_key_secret:
            api_key = await self._context.resources.secret_value(config.resource, "api_key_secret")
        if resource.dsn_secret:
            dsn = await self._context.resources.secret_value(config.resource, "dsn_secret")
        reader = await reader_for(resource, api_key=api_key, dsn=dsn)
        documents = await reader.search(
            config.collection,
            vector,
            config.top_k,
            config.filter,
            min_score=config.min_score,
        )
        return {f"{node.id}.documents": documents}

    async def _run_rerank(
        self, node: RerankNode, inputs: dict[str, PortValue]
    ) -> dict[str, PortValue]:
        config = node.config
        query = _as_text(inputs.get("query", ""))
        docs_value = inputs.get("documents")
        documents = (
            [d for d in docs_value if isinstance(d, Document)]
            if isinstance(docs_value, list)
            else []
        )
        if not documents:
            return {f"{node.id}.documents": []}
        client, default_model = await self._llm_client(config.resource)
        model = config.model or default_model
        if not model:
            raise FlowError(f"node {node.id!r}: no rerank model configured")
        ranked = await client.rerank(model, query, [doc.text for doc in documents], config.top_n)
        reranked: list[Document] = []
        for index, score in ranked:
            if not 0 <= index < len(documents):
                continue
            if config.min_score is not None and score < config.min_score:
                continue
            original = documents[index]
            reranked.append(Document(text=original.text, score=score, metadata=original.metadata))
        return {f"{node.id}.documents": reranked[: config.top_n]}

    async def _run_mcp_tool(
        self, node: McpToolNode, inputs: dict[str, PortValue]
    ) -> dict[str, PortValue]:
        from fastmcp import Client  # noqa: PLC0415 - deferred: heavy import

        config = node.config
        url = config.url
        auth: str | None = None
        if config.resource:
            resource = await self._context.resources.get_raw(config.resource)
            if not isinstance(resource, McpServerResource):
                raise FlowError(f"resource {config.resource!r} is not an MCP server")
            url = resource.url
            if resource.auth_secret:
                auth = await self._context.resources.secret_value(config.resource, "auth_secret")
        if not url:
            raise FlowError(f"node {node.id!r}: no MCP server url")
        arguments = {arg_name: _as_json(inputs.get(port)) for port, arg_name in config.args.items()}
        client = Client(url, auth=auth) if auth else Client(url)
        async with client:
            result = await client.call_tool(config.tool, arguments)
        return {f"{node.id}.result": _tool_result_to_json(result)}

    async def _run_agent(
        self, node: AgentNode, inputs: dict[str, PortValue]
    ) -> dict[str, PortValue]:
        config = node.config
        rendered = {port: _as_text(value) for port, value in inputs.items()}
        for port in input_ports(node):
            rendered.setdefault(port, "")
        prompt = _format_prompt(config.prompt, rendered)
        system_prompt = _format_prompt(config.system_prompt, rendered)
        client, default_model = await self._llm_client(config.resource)
        model = config.model or default_model
        if not model:
            raise FlowError(f"node {node.id!r}: no model configured (node or resource default)")

        tool_schemas, targets, agent_targets = await self._agent_tools(config)
        messages: list[dict[str, object]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        final_text = ""
        for _ in range(config.max_iterations):
            message = await client.chat_with_tools(model, messages, tool_schemas)
            messages.append(message)
            calls = message.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                final_text = str(message.get("content") or "")
                break
            # One turn's tool calls run concurrently — an orchestrator asking
            # several sub-agents pays only the slowest one, not the sum.
            contents = await asyncio.gather(
                *(self._call_agent_tool(call, targets, agent_targets) for call in calls)
            )
            for call, content in zip(calls, contents, strict=True):
                call_id = call.get("id", "") if isinstance(call, dict) else ""
                messages.append({"role": "tool", "tool_call_id": str(call_id), "content": content})
        else:
            # ran out of tool-call turns — force one final answer without tools
            message = await client.chat_with_tools(model, messages, [])
            final_text = str(message.get("content") or "")
        return {f"{node.id}.text": final_text}

    async def _sub_agent_tools(
        self, config: AgentNodeConfig
    ) -> tuple[list[dict[str, object]], dict[str, ResolvedAgent]]:
        """One tool per referenced registry agent; description from its card."""
        schemas: list[dict[str, object]] = []
        agent_targets: dict[str, ResolvedAgent] = {}
        for ref in config.agents:
            if self._context.agents is None:
                raise FlowError("agent references require a configured registry")
            resolved = await self._context.agents.resolve(ref.name)
            tool_name = f"agent-{resolved.name}"
            description = resolved.description or f"Delegate to the {resolved.name} agent."
            if resolved.examples:
                description += " Example requests: " + "; ".join(resolved.examples[:3])
            agent_targets[tool_name] = resolved
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "The request to send to this agent.",
                                }
                            },
                            "required": ["message"],
                        },
                    },
                }
            )
        return schemas, agent_targets

    async def _agent_tools(
        self, config: AgentNodeConfig
    ) -> tuple[list[dict[str, object]], dict[str, _ToolTarget], dict[str, ResolvedAgent]]:
        """Resolve tool + sub-agent refs to OpenAI tool schemas and target maps."""
        schemas, agent_targets = await self._sub_agent_tools(config)
        targets: dict[str, _ToolTarget] = {}

        if not config.tools:
            return schemas, targets, agent_targets
        from fastmcp import Client  # noqa: PLC0415 - deferred: heavy import

        wanted: dict[str, set[str] | None] = {}  # None = every tool on that server
        for tool_ref in config.tools:
            if not tool_ref.tool:
                wanted[tool_ref.resource] = None
            else:
                existing = wanted.get(tool_ref.resource, set())
                if existing is not None:
                    existing.add(tool_ref.tool)
                    wanted[tool_ref.resource] = existing

        for resource_name, only in wanted.items():
            resource = await self._context.resources.get_raw(resource_name)
            if not isinstance(resource, McpServerResource):
                raise FlowError(f"resource {resource_name!r} is not an MCP server")
            auth: str | None = None
            if resource.auth_secret:
                auth = await self._context.resources.secret_value(resource_name, "auth_secret")
            client = Client(resource.url, auth=auth) if auth else Client(resource.url)
            async with client:
                available = await client.list_tools()
            for tool in available:
                if only is not None and tool.name not in only:
                    continue
                if tool.name in agent_targets:
                    continue  # sub-agent tools keep their name; skip the MCP shadow
                if tool.name in targets:
                    continue  # another server already exposed this name; no duplicate schema
                targets[tool.name] = _ToolTarget(url=resource.url, auth=auth, tool=tool.name)
                schemas.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                        },
                    }
                )
        return schemas, targets, agent_targets

    async def _call_agent_tool(
        self,
        call: object,
        targets: dict[str, _ToolTarget],
        agent_targets: dict[str, ResolvedAgent],
    ) -> str:
        function = call.get("function") if isinstance(call, dict) else None
        name = str(function.get("name", "")) if isinstance(function, dict) else ""
        raw_args = function.get("arguments", "{}") if isinstance(function, dict) else "{}"
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except ValueError:
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        agent = agent_targets.get(name)
        if agent is not None:
            return await self._call_a2a_agent(agent, _as_text(arguments.get("message") or ""))

        target = targets.get(name)
        if target is None:
            return f"error: unknown tool {name!r}"
        from fastmcp import Client  # noqa: PLC0415 - deferred: heavy import

        # Match the sub-agent contract: a failing tool comes back as an `error:`
        # result the model can react to, instead of aborting the whole turn (and
        # discarding sibling tool/sub-agent calls running in the same gather).
        try:
            client = Client(target.url, auth=target.auth) if target.auth else Client(target.url)
            async with client:
                result = await client.call_tool(target.tool, arguments)
        except Exception as exc:  # returned to the model as an error result, never raised
            return f"error: tool {name!r} failed: {exc}"
        return _as_text(_tool_result_to_json(result))

    async def _call_a2a_agent(self, agent: ResolvedAgent, message: str) -> str:
        """Delegate one request to a registry agent over A2A (through the gateway).

        Returns the sub-agent's final artifact text; failures come back as
        ``error: ...`` tool results so the orchestrating model can react. The
        call depth travels in the message metadata and is enforced on receipt
        by every runtime — the sender-side check just saves a doomed call.
        """
        from uuid import uuid4  # noqa: PLC0415 - deferred with its sole consumer

        from a2a.client import ClientConfig, ClientFactory, minimal_agent_card  # noqa: PLC0415
        from a2a.types.a2a_pb2 import (  # noqa: PLC0415 - deferred: heavy import
            Message,
            Part,
            Role,
            SendMessageRequest,
            TaskState,
        )

        depth = _call_depth_var.get() + 1
        if depth > self._context.settings.max_agent_call_depth:
            return (
                f"error: agent call depth limit "
                f"({self._context.settings.max_agent_call_depth}) exceeded"
            )
        headers: dict[str, str] = {}
        # Propagate the active trace (SPEC §12): the sub-agent's flow span joins
        # this trace, so a multi-agent chain reads as ONE trace end to end.
        propagate.inject(headers)
        # Token passthrough only to same-gateway agents: a registry URL not under
        # our public/gateway base must never receive the caller's JWT — a squatted
        # or foreign entry would otherwise exfiltrate the credential.
        base = self._context.settings.public_base_url
        if (token := caller_token_var.get()) and (not base or _same_origin(agent.url, base)):
            headers["Authorization"] = f"Bearer {token}"
        outgoing = Message(message_id=str(uuid4()), role=Role.ROLE_USER, parts=[Part(text=message)])
        outgoing.metadata.update({CALL_DEPTH_METADATA_KEY: depth})

        final_task = None
        reply_text = ""
        try:
            async with httpx.AsyncClient(
                timeout=self._context.settings.http_timeout_s,
                headers=headers,
                follow_redirects=True,
            ) as http_client:
                factory = ClientFactory(ClientConfig(streaming=False, httpx_client=http_client))
                # The JSON-RPC route lives at "/" under the endpoint prefix;
                # the slash avoids a 307 from the serving router.
                card = minimal_agent_card(agent.url.rstrip("/") + "/", ["JSONRPC"])
                client = factory.create(card)
                async for event in client.send_message(SendMessageRequest(message=outgoing)):
                    if event.HasField("task"):
                        final_task = event.task
                    elif event.HasField("message"):
                        reply_text = "".join(p.text for p in event.message.parts if p.text)
        except Exception as exc:
            return f"error: agent {agent.name!r} unreachable: {exc}"

        if final_task is None:
            return reply_text or f"error: agent {agent.name!r} returned no result"
        if final_task.status.state == TaskState.TASK_STATE_COMPLETED:
            return "".join(
                part.text
                for artifact in final_task.artifacts
                for part in artifact.parts
                if part.text
            )
        state_name = TaskState.Name(final_task.status.state)
        status_text = "".join(p.text for p in final_task.status.message.parts if p.text)
        detail = f": {status_text}" if status_text else ""
        return f"error: agent {agent.name!r} ended in {state_name}{detail}"

    async def execute(
        self,
        start_values: JsonObject,
        *,
        stream: StreamCallback | None = None,
        trace_attributes: Mapping[str, str] | None = None,
        conversation: Sequence[ConversationTurn] | None = None,
        call_depth: int = 0,
    ) -> PortValue:
        """Run the flow; returns the value of the end node's output.

        ``trace_attributes`` are added to the flow span — the caller's user
        and conversation attribution (`user.id`, `session.id`, SPEC §12).
        ``conversation`` is the caller's prior turns; LLM nodes with
        ``history: true`` prepend them as chat messages. ``call_depth`` is the
        orchestration depth of the incoming request (from the A2A message
        metadata); outgoing sub-agent calls carry ``call_depth + 1``.
        """
        token = _stream_var.set(stream)
        conversation_token = _conversation_var.set(tuple(conversation or ()))
        depth_token = _call_depth_var.set(call_depth)
        try:
            return await self._execute(start_values, trace_attributes or {})
        finally:
            _call_depth_var.reset(depth_token)
            _conversation_var.reset(conversation_token)
            _stream_var.reset(token)

    async def _execute(
        self, start_values: JsonObject, trace_attributes: Mapping[str, str]
    ) -> PortValue:
        start = next(n for n in self._defn.nodes if isinstance(n, StartNode))
        end = next(n for n in self._defn.nodes if isinstance(n, EndNode))
        seeded: dict[str, PortValue] = {
            f"{start.id}.{key}": value for key, value in start_values.items()
        }
        input_preview = _trace_preview(start_values)
        with tracer.start_as_current_span(
            "agentplane.flow",
            attributes={
                "flow.name": self._context.flow_name,
                "flow.version": self._context.flow_version,
                # What was asked / answered, in OpenInference terms (tracing
                # UIs map input.value/output.value to observation input/output).
                "input.value": input_preview,
                # Trace-level overrides: the flow IS the trace (the HTTP spans
                # around it are dropped in the collector), so name the trace
                # after the flow and surface the exchange at trace level.
                # Plain OTLP attribute hints - backends that don't know them
                # simply ignore them (no vendor SDK involved).
                "langfuse.trace.name": self._context.flow_name,
                "langfuse.trace.input": input_preview,
                **trace_attributes,
            },
        ) as span:
            graph = self._graph
            empty: set[str] = set()
            result = await graph.ainvoke(  # type: ignore[attr-defined]
                {"values": seeded, "executed": empty}
            )
            values: dict[str, PortValue] = result["values"]
            output = values.get(f"{end.id}.output")
            if output is not None:
                output_preview = _trace_preview(output)
                span.set_attribute("output.value", output_preview)
                span.set_attribute("langfuse.trace.output", output_preview)
        return output


_TRACE_PREVIEW_MAX = 4000


def _trace_preview(value: object) -> str:
    """Span-attribute rendering of an input/output value, length-capped.

    A single-string input object (the common chat case) renders as the bare
    text; everything else as compact JSON.
    """
    if isinstance(value, dict) and len(value) == 1:
        only = next(iter(value.values()))
        if isinstance(only, str):
            value = only
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text[:_TRACE_PREVIEW_MAX]


def _is_empty(value: PortValue | None) -> bool:
    """Router emptiness: None, blank text, empty list/object count as empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _resolve_path(value: PortValue | None, path: str) -> PortValue | None:
    """Follow a dot path into dicts (and numeric segments into lists); miss -> None."""
    # `object` because list elements widen the union (e.g. Document in a list);
    # every reachable value is still a valid port value.
    current: object = value
    for segment in path.split("."):
        if isinstance(current, dict):
            current = current.get(segment)
        elif isinstance(current, list) and segment.isdigit() and int(segment) < len(current):
            current = current[int(segment)]
        else:
            return None
    return cast("PortValue | None", current)


def _is_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _compare_numeric(when: str, target: float, operand: float) -> bool:
    match when:
        case "gt":
            return target > operand
        case "gte":
            return target >= operand
        case "lt":
            return target < operand
        case _:
            return target <= operand


def _rule_matches(rule: RouterRule, value: PortValue | None) -> bool:  # noqa: PLR0911
    """One condition against the (path-resolved) input value."""
    target = _resolve_path(value, rule.path) if rule.path else value
    match rule.when:
        case "empty":
            return _is_empty(target)
        case "not_empty":
            return not _is_empty(target)
        case "equals":
            return target == rule.value
        case "not_equals":
            # A missing/null path never matches a comparison (RouterRule contract).
            return target is not None and target != rule.value
        case "contains":
            if isinstance(target, str) and isinstance(rule.value, str):
                return rule.value in target
            if isinstance(target, list):
                return rule.value in target
            if isinstance(target, dict) and isinstance(rule.value, str):
                return rule.value in target
            return False
        case _:
            if _is_number(target) and _is_number(rule.value):
                return _compare_numeric(rule.when, float(target), float(rule.value))
            return False


def _route(config: RouterNodeConfig, value: PortValue | None) -> str:
    """First matching rule wins; otherwise the default branch."""
    for rule in config.rules:
        if _rule_matches(rule, value):
            return rule.branch
    return config.default_branch


def _same_origin(url: str, base: str) -> bool:
    """True when ``url`` shares scheme+host+port with ``base`` (the gateway origin)."""
    a, b = urlsplit(url), urlsplit(base)
    return (a.scheme, a.hostname, a.port) == (b.scheme, b.hostname, b.port)


def _format_prompt(template: str, values: dict[str, str]) -> str:
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace("{" + name + "}", value)
    return rendered.replace("{{", "{").replace("}}", "}")


def _tool_result_to_json(result: object) -> JsonValue:
    """Map a FastMCP CallToolResult to a JSON port value."""
    data = getattr(result, "data", None)
    if data is not None:
        try:
            dumped = json.dumps(data, default=str)
        except (TypeError, ValueError):
            return str(data)
        loaded: JsonValue = json.loads(dumped)
        return loaded
    content = getattr(result, "content", None)
    if isinstance(content, list) and content:
        first = content[0]
        text = getattr(first, "text", None)
        if isinstance(text, str):
            return text
    return None


__all__ = [
    "CALL_DEPTH_METADATA_KEY",
    "ExecutionContext",
    "FlowError",
    "FlowRunner",
    "FlowState",
    "PortValue",
    "StreamCallback",
    "caller_token_var",
]
