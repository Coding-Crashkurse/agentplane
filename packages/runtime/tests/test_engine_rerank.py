"""Rerank node: the /rerank client mapping + the executor reorder/threshold."""

from __future__ import annotations

import httpx
import respx
from cryptography.fernet import Fernet

from agentplane_core import Document, ModelProviderResource, RerankNode
from agentplane_runtime.db import Database
from agentplane_runtime.engine import ExecutionContext, FlowRunner
from agentplane_runtime.llm import OpenAICompatibleClient
from agentplane_runtime.resources import ResourceService
from agentplane_runtime.secrets import FernetSecretsProvider

from .conftest import LLM_BASE, load_example, make_settings


@respx.mock
async def test_rerank_client_maps_results_to_sorted_pairs() -> None:
    respx.post(f"{LLM_BASE}/rerank").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.4},
                ]
            },
        )
    )
    client = OpenAICompatibleClient(LLM_BASE)
    ranked = await client.rerank("rerank-model", "q", ["a", "b", "c"], top_n=2)
    assert ranked == [(2, 0.9), (0, 0.4)]


async def _rerank_runner() -> tuple[FlowRunner, RerankNode]:
    db = Database("sqlite+aiosqlite://")
    await db.create_all()
    secrets = FernetSecretsProvider(db, Fernet.generate_key().decode("ascii"))
    resources = ResourceService(db, secrets)
    await resources.create(
        ModelProviderResource(name="reranker", base_url=LLM_BASE, default_model="rerank-model"),
        "anonymous",
    )
    defn = load_example("rag-with-rerank.yaml")
    node = next(n for n in defn.nodes if isinstance(n, RerankNode))
    runner = FlowRunner(
        defn,
        ExecutionContext(
            resources=resources, settings=make_settings(), flow_name=defn.name, flow_version=1
        ),
    )
    return runner, node


@respx.mock
async def test_run_rerank_reorders_by_score() -> None:
    respx.post(f"{LLM_BASE}/rerank").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.80},
                    {"index": 2, "relevance_score": 0.10},
                ]
            },
        )
    )
    runner, node = await _rerank_runner()
    docs = [Document(text="alpha"), Document(text="beta"), Document(text="gamma")]
    out = await runner._run_rerank(node, {"query": "q", "documents": docs})
    reranked = out["rerank_1.documents"]
    assert isinstance(reranked, list)
    ordered = [d.text for d in reranked if isinstance(d, Document)]
    assert ordered == ["beta", "alpha", "gamma"]
    assert isinstance(reranked[0], Document)
    assert reranked[0].score == 0.95


async def test_run_rerank_short_circuits_on_empty_documents() -> None:
    runner, node = await _rerank_runner()
    out = await runner._run_rerank(node, {"query": "q", "documents": []})
    assert out == {"rerank_1.documents": []}
