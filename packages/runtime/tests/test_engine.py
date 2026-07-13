"""Engine unit tests: prompt binding, streaming, structured output, tracing-safe paths."""

from __future__ import annotations

import httpx
import pytest
import respx
from cryptography.fernet import Fernet

from agentplane_core import Document, ModelProviderResource, render_documents
from agentplane_runtime.db import Database
from agentplane_runtime.engine import ExecutionContext, FlowError, FlowRunner, _format_prompt
from agentplane_runtime.resources import ResourceService
from agentplane_runtime.secrets import FernetSecretsProvider

from .conftest import LLM_BASE, load_example, make_settings


async def _runner(defn_name: str = "echo-agent.yaml") -> FlowRunner:
    db = Database("sqlite+aiosqlite://")
    await db.create_all()
    secrets = FernetSecretsProvider(db, Fernet.generate_key().decode("ascii"))
    resources = ResourceService(db, secrets)
    await resources.create(
        ModelProviderResource(
            name="default-llm",
            base_url=LLM_BASE,
            api_key_secret="sk-super-secret-abcdef",
            default_model="gpt-5-mini",
        ),
        "anonymous",
    )
    defn = load_example(defn_name)
    return FlowRunner(
        defn,
        ExecutionContext(
            resources=resources, settings=make_settings(), flow_name=defn.name, flow_version=1
        ),
    )


@respx.mock
async def test_execute_returns_end_output() -> None:
    route = respx.post(f"{LLM_BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": "pong"}}]})
    )
    runner = await _runner()
    result = await runner.execute({"message": "ping"})
    assert result == "pong"
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer sk-super-secret-abcdef"
    body = request.content.decode()
    assert '"ping"' in body or "ping" in body


@respx.mock
async def test_streaming_callback_receives_deltas() -> None:
    sse = (
        'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(f"{LLM_BASE}/chat/completions").mock(
        return_value=httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
    )
    runner = await _runner()
    seen: list[str] = []

    async def collect(delta: str) -> None:
        seen.append(delta)

    result = await runner.execute({"message": "ping"}, stream=collect)
    assert result == "ab"
    assert seen == ["a", "b"]


@respx.mock
async def test_llm_error_raises_flow_error() -> None:
    respx.post(f"{LLM_BASE}/chat/completions").mock(return_value=httpx.Response(503))
    runner = await _runner()
    with pytest.raises(Exception, match="chat completion"):
        await runner.execute({"message": "ping"})


async def test_missing_model_raises_flow_error() -> None:
    db = Database("sqlite+aiosqlite://")
    await db.create_all()
    secrets = FernetSecretsProvider(db, Fernet.generate_key().decode("ascii"))
    resources = ResourceService(db, secrets)
    await resources.create(
        ModelProviderResource(name="default-llm", base_url=LLM_BASE), "anonymous"
    )
    defn = load_example("echo-agent.yaml")
    runner = FlowRunner(defn, ExecutionContext(resources=resources, settings=make_settings()))
    with pytest.raises(FlowError, match="no model"):
        await runner.execute({"message": "ping"})


def test_format_prompt_replaces_variables() -> None:
    rendered = _format_prompt(
        "Answer using: {documents}\n\nQ: {query}",
        {
            "documents": "doc-text",
            "query": "how?",
        },
    )
    assert rendered == "Answer using: doc-text\n\nQ: how?"


def test_format_prompt_keeps_double_braces() -> None:
    assert _format_prompt('{{"json": true}}', {}) == '{"json": true}'


def test_render_documents_includes_source_headers() -> None:
    documents = [
        Document(text="chunk one", metadata={"source": "kb/1"}),
        Document(text="chunk two"),
    ]
    rendered = render_documents(documents)
    assert "[kb/1]" in rendered
    assert "chunk one" in rendered
    assert "[document 2]" in rendered
