"""Fan-in joins: every node runs at most once, and router branches gate execution.

LangGraph triggers a node as soon as *any* predecessor wrote, so a node fed
from different graph depths (``start -> call`` AND ``start -> retrieve ->
call``) used to run twice — the first time with its documents port still empty,
i.e. a wasted LLM call. In a routed flow it was worse: an LLM node that also
takes an unconditional edge from ``start`` ran even when the router had chosen
the other branch, and its output won the race on ``end.input``.
"""

from __future__ import annotations

from collections import Counter

import httpx
import respx
from cryptography.fernet import Fernet

from agentplane_core import FlowDefinition, ModelProviderResource, Node, VectorDBResource
from agentplane_runtime.db import Database
from agentplane_runtime.engine import ExecutionContext, FlowRunner, PortValue
from agentplane_runtime.resources import ResourceService
from agentplane_runtime.secrets import FernetSecretsProvider

from .conftest import LLM_BASE, QDRANT_BASE, load_example, make_settings, vector_db_body

SEARCH_URL = f"{QDRANT_BASE}/collections/support_docs/points/search"


def _count_executions(runner: FlowRunner) -> Counter[str]:
    """Spy on the per-node executor: it runs a node's body only when the node
    actually fires (the readiness/dedup guard skips re-triggers without calling
    it), so the returned counter is the true per-node execution count."""
    counts: Counter[str] = Counter()
    original = runner._run

    async def counting(node: Node, values: dict[str, PortValue]) -> dict[str, PortValue]:
        counts[node.id] += 1
        return await original(node, values)

    runner._run = counting  # type: ignore[method-assign]
    return counts


async def _runner(defn: FlowDefinition) -> FlowRunner:
    db = Database("sqlite+aiosqlite://")
    await db.create_all()
    resources = ResourceService(db, FernetSecretsProvider(db, Fernet.generate_key().decode()))
    await resources.create(
        ModelProviderResource(name="default-llm", base_url=LLM_BASE, default_model="gpt-5-mini"),
        "anonymous",
    )
    await resources.create(VectorDBResource.model_validate(vector_db_body()), "anonymous")
    return FlowRunner(
        defn,
        ExecutionContext(resources=resources, settings=make_settings(), flow_name=defn.name),
    )


def _mock_embeddings() -> None:
    respx.post(f"{LLM_BASE}/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]})
    )


def _mock_search(*documents: dict[str, object]) -> respx.Route:
    return respx.post(SEARCH_URL).mock(return_value=httpx.Response(200, json={"result": documents}))


def _mock_llm(answer: str) -> respx.Route:
    return respx.post(f"{LLM_BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})
    )


@respx.mock
async def test_fan_in_node_runs_once() -> None:
    """support-rag feeds `call_1` from `start_1` AND `retrieve_1` — one LLM call, not two."""
    _mock_embeddings()
    _mock_search({"score": 0.9, "payload": {"text": "Reset the router.", "source": "kb/1"}})
    llm = _mock_llm("Please reset the router.")

    runner = await _runner(load_example("support-rag.yaml"))
    result = await runner.execute({"query": "wifi broken"})

    assert result == "Please reset the router."
    assert llm.call_count == 1
    # the single call saw the retrieved documents — it did not fire before retrieval
    assert "Reset the router." in llm.calls.last.request.content.decode()


@respx.mock
async def test_router_gates_the_llm_branch() -> None:
    """Nothing retrieved -> the LLM node must not run at all, fallback text is returned."""
    _mock_embeddings()
    _mock_search()  # empty result set
    llm = _mock_llm("should never be produced")

    runner = await _runner(load_example("rag-with-fallback.yaml"))
    result = await runner.execute({"query": "unknown topic"})

    assert result == "I have no information about that in the knowledge base."
    assert llm.call_count == 0


@respx.mock
async def test_router_found_branch_runs_the_llm_once() -> None:
    _mock_embeddings()
    _mock_search({"score": 0.8, "payload": {"text": "Reset the router.", "source": "kb/1"}})
    llm = _mock_llm("Reset it.")

    runner = await _runner(load_example("rag-with-fallback.yaml"))
    result = await runner.execute({"query": "wifi broken"})

    assert result == "Reset it."
    assert llm.call_count == 1


@respx.mock
async def test_fan_in_executes_each_node_exactly_once() -> None:
    """Direct execution-count pin: `call_1` fans in from `start_1` AND `retrieve_1`
    (a join across two graph depths), yet every node — including the shared LLM
    node — executes exactly once (SPEC §6.4)."""
    _mock_embeddings()
    _mock_search({"score": 0.9, "payload": {"text": "Reset the router.", "source": "kb/1"}})
    _mock_llm("Please reset the router.")

    runner = await _runner(load_example("support-rag.yaml"))
    counts = _count_executions(runner)
    await runner.execute({"query": "wifi broken"})

    assert counts == Counter({"start_1": 1, "retrieve_1": 1, "call_1": 1, "end_1": 1})


@respx.mock
async def test_router_untaken_branch_node_stays_dormant() -> None:
    """Nothing retrieved -> router picks the `missing` branch; the LLM node on the
    untaken `found` branch never executes, and no node runs twice (SPEC §6.4)."""
    _mock_embeddings()
    _mock_search()  # empty result set -> the `empty`/default branch fires
    _mock_llm("should never be produced")

    runner = await _runner(load_example("rag-with-fallback.yaml"))
    counts = _count_executions(runner)
    result = await runner.execute({"query": "unknown topic"})

    assert result == "I have no information about that in the knowledge base."
    assert counts["call_1"] == 0  # untaken branch stays dormant
    assert counts == Counter(
        {"start_1": 1, "retrieve_1": 1, "check_1": 1, "fallback_1": 1, "end_1": 1}
    )


@respx.mock
async def test_router_found_branch_runs_llm_once_fallback_dormant() -> None:
    """Documents retrieved -> the `found` branch runs the LLM once; the fallback
    template on the untaken branch stays dormant (SPEC §6.4)."""
    _mock_embeddings()
    _mock_search({"score": 0.8, "payload": {"text": "Reset the router.", "source": "kb/1"}})
    _mock_llm("Reset it.")

    runner = await _runner(load_example("rag-with-fallback.yaml"))
    counts = _count_executions(runner)
    result = await runner.execute({"query": "wifi broken"})

    assert result == "Reset it."
    assert counts["fallback_1"] == 0  # untaken branch stays dormant
    assert counts == Counter({"start_1": 1, "retrieve_1": 1, "check_1": 1, "call_1": 1, "end_1": 1})


@respx.mock
async def test_min_score_is_sent_to_the_vector_db() -> None:
    """Without a threshold a filled collection always returns hits (FEEDBACK 2.1)."""
    _mock_embeddings()
    search = _mock_search()
    _mock_llm("unused")

    runner = await _runner(load_example("rag-with-fallback.yaml"))
    await runner.execute({"query": "unknown topic"})

    assert search.calls.last.request.read().decode().count('"score_threshold":0.5') == 1
