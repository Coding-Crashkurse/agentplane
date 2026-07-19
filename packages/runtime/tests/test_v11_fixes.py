"""Regression tests for the v1.1 node fixes (adversarial-review findings)."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from agentplane_core import (
    AgentNodeConfig,
    AgentToolRef,
    McpServerResource,
    ModelProviderResource,
    RouterNodeConfig,
    RouterRule,
)
from agentplane_runtime.db import Database
from agentplane_runtime.engine import (
    ExecutionContext,
    FlowRunner,
    _route,
    _same_origin,
    _ToolTarget,
)
from agentplane_runtime.resources import ResourceService
from agentplane_runtime.secrets import FernetSecretsProvider

from .conftest import load_example, make_settings


def test_router_not_equals_missing_path_falls_through() -> None:
    """Fix 1: not_equals must never match a missing/null path (RouterRule contract)."""
    config = RouterNodeConfig(
        input_type="json",
        rules=[RouterRule(when="not_equals", path="category", value="spam", branch="keep")],
        default_branch="otherwise",
    )
    assert _route(config, {"category": "ham"}) == "keep"  # present + different -> matches
    assert _route(config, {"other": "x"}) == "otherwise"  # path absent -> must NOT match
    assert _route(config, {"category": "spam"}) == "otherwise"  # equal -> not_equals false


def test_same_origin_gates_token_passthrough() -> None:
    """Fix 4: the caller token only rides to same-gateway (same-origin) sub-agents."""
    base = "https://api.example"
    assert _same_origin("https://api.example/a2a/echo", base)
    assert not _same_origin("https://evil.example/a2a/echo", base)  # foreign host
    assert not _same_origin("http://api.example/a2a/echo", base)  # scheme differs
    assert not _same_origin("https://api.example:8443/a2a/echo", base)  # port differs


class _FakeMcpTool:
    def __init__(self, name: str) -> None:
        self.name = name
        self.description = ""
        self.inputSchema: dict[str, object] = {"type": "object", "properties": {}}


class _CollidingMcpClient:
    """Every server exposes a tool named 'search' — a cross-server name collision."""

    def __init__(self, url: str, auth: str | None = None) -> None:
        self._url = url

    async def __aenter__(self) -> _CollidingMcpClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def list_tools(self) -> list[_FakeMcpTool]:
        return [_FakeMcpTool("search")]


class _RaisingMcpClient:
    """call_tool raises, as a transient MCP 500/connection-reset would."""

    def __init__(self, url: str, auth: str | None = None) -> None:
        self._url = url

    async def __aenter__(self) -> _RaisingMcpClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def call_tool(self, name: str, args: dict[str, object]) -> object:
        raise RuntimeError("mcp server 500")


async def _runner() -> FlowRunner:
    db = Database("sqlite+aiosqlite://")
    await db.create_all()
    secrets = FernetSecretsProvider(db, Fernet.generate_key().decode("ascii"))
    resources = ResourceService(db, secrets)
    await resources.create(
        ModelProviderResource(name="default-llm", base_url="http://llm.test", default_model="m"),
        "anonymous",
    )
    await resources.create(McpServerResource(name="srv-a", url="http://a.test/mcp"), "anonymous")
    await resources.create(McpServerResource(name="srv-b", url="http://b.test/mcp"), "anonymous")
    defn = load_example("agent-with-tools.yaml")
    return FlowRunner(
        defn, ExecutionContext(resources=resources, settings=make_settings(), flow_name=defn.name)
    )


async def test_agent_tools_dedup_colliding_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix 2: two servers exposing the same tool name -> one schema, not a 400."""
    import fastmcp  # noqa: PLC0415 - patched for this test only

    monkeypatch.setattr(fastmcp, "Client", _CollidingMcpClient)
    runner = await _runner()
    config = AgentNodeConfig(
        resource="default-llm",
        prompt="{message}",
        tools=[AgentToolRef(resource="srv-a"), AgentToolRef(resource="srv-b")],
    )
    schemas, targets, _agents = await runner._agent_tools(config)
    names: list[object] = []
    for schema in schemas:
        function = schema.get("function")
        if isinstance(function, dict):
            names.append(function.get("name"))
    assert names.count("search") == 1
    assert set(targets) == {"search"}


async def test_agent_mcp_tool_error_becomes_error_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix 5: a failing MCP tool returns an `error:` result, never aborts the turn."""
    import fastmcp  # noqa: PLC0415 - patched for this test only

    monkeypatch.setattr(fastmcp, "Client", _RaisingMcpClient)
    runner = await _runner()
    targets = {"search": _ToolTarget(url="http://a.test/mcp", auth=None, tool="search")}
    call: dict[str, object] = {"function": {"name": "search", "arguments": "{}"}}
    result = await runner._call_agent_tool(call, targets, {})
    assert result.startswith("error: tool 'search' failed")
