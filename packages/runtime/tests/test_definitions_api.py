"""Definitions API and draft/version semantics (SPEC §6.1/§6.2)."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import httpx
import respx
from asgi_lifespan import LifespanManager

from agentplane_runtime.app import create_app

from .conftest import (
    QDRANT_BASE,
    REGISTRY_BASE,
    create_default_resources,
    load_example,
    make_settings,
)


async def test_validate_endpoint_reports_stateful_codes(client: httpx.AsyncClient) -> None:
    defn = load_example("echo-agent.yaml").canonical_dict()
    # no resources registered yet -> E020, still HTTP 200 (SPEC §6.1)
    response = await client.post("/api/v1/definitions/validate", json=defn)
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is False
    assert [i["code"] for i in body["issues"]] == ["E020"]

    await create_default_resources(client)
    body = (await client.post("/api/v1/definitions/validate", json=defn)).json()
    assert body["valid"] is True


async def test_validate_reports_e021_kind_mismatch(client: httpx.AsyncClient) -> None:
    await create_default_resources(client)
    defn = load_example("echo-agent.yaml").canonical_dict()
    nodes = defn["nodes"]
    assert isinstance(nodes, list)
    for node in nodes:
        assert isinstance(node, dict)
        if node["type"] == "llm_call":
            config = node["config"]
            assert isinstance(config, dict)
            config["resource"] = "kb-support"  # vector DB used as model provider
    body = (await client.post("/api/v1/definitions/validate", json=defn)).json()
    assert "E021" in [i["code"] for i in body["issues"]]


async def test_create_draft_invalid_returns_422_with_result(client: httpx.AsyncClient) -> None:
    defn = load_example("echo-agent.yaml").canonical_dict()
    response = await client.post("/api/v1/definitions", json=defn)  # E020: no resources
    assert response.status_code == 422
    assert response.json()["issues"][0]["code"] == "E020"


async def test_draft_lifecycle(client: httpx.AsyncClient) -> None:
    await create_default_resources(client)
    defn = load_example("echo-agent.yaml")
    created = await client.post("/api/v1/definitions", json=defn.canonical_dict())
    assert created.status_code == 201
    info = created.json()
    assert info["status"] == "draft"
    assert info["latest_version"] is None

    # duplicate name -> 409
    assert (await client.post("/api/v1/definitions", json=defn.canonical_dict())).status_code == 409

    # update draft
    updated_defn = defn.model_copy(update={"description": "now with more echo"})
    updated = await client.put(
        f"/api/v1/definitions/{defn.name}", json=updated_defn.canonical_dict()
    )
    assert updated.status_code == 200
    assert updated.json()["description"] == "now with more echo"

    listing = (await client.get("/api/v1/definitions")).json()
    assert [d["name"] for d in listing] == ["echo-agent"]

    detail = (
        await client.get("/api/v1/definitions/echo-agent", params={"include": "definition"})
    ).json()
    assert detail["definition"]["name"] == "echo-agent"


async def test_deploy_versions_and_rollback(client: httpx.AsyncClient) -> None:
    await create_default_resources(client)
    defn = load_example("echo-agent.yaml")
    await client.post("/api/v1/definitions", json=defn.canonical_dict())

    first = await client.post("/api/v1/definitions/echo-agent/deploy")
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["version"] == 1
    assert body["endpoint_url"] == "https://api.example/a2a/echo-agent"

    # draft edits do not touch the deployed version until the next deploy
    updated = defn.model_copy(update={"description": "v2"})
    await client.put("/api/v1/definitions/echo-agent", json=updated.canonical_dict())
    second = await client.post("/api/v1/definitions/echo-agent/deploy")
    assert second.json()["version"] == 2

    info = (await client.get("/api/v1/definitions/echo-agent")).json()
    assert info["status"] == "deployed"
    assert info["deployed_version"] == 2
    assert info["latest_version"] == 2

    # rollback re-serves version 1 without creating a new version
    rollback = await client.post("/api/v1/definitions/echo-agent/deploy", params={"version": 1})
    assert rollback.json()["version"] == 1
    info = (await client.get("/api/v1/definitions/echo-agent")).json()
    assert info["deployed_version"] == 1
    assert info["latest_version"] == 2

    # exported version 2 carries the draft edit; version 1 does not
    v2 = (await client.get("/api/v1/definitions/echo-agent/export", params={"version": 2})).json()
    assert v2["description"] == "v2"
    v1 = (await client.get("/api/v1/definitions/echo-agent/export", params={"version": 1})).json()
    assert v1["description"] != "v2"


async def test_undeploy_and_delete(client: httpx.AsyncClient) -> None:
    await create_default_resources(client)
    defn = load_example("echo-agent.yaml")
    await client.post("/api/v1/definitions", json=defn.canonical_dict())
    await client.post("/api/v1/definitions/echo-agent/deploy")

    # deployed definitions cannot be deleted
    assert (await client.delete("/api/v1/definitions/echo-agent")).status_code == 409
    assert (await client.post("/api/v1/definitions/echo-agent/undeploy")).status_code == 204
    info = (await client.get("/api/v1/definitions/echo-agent")).json()
    assert info["status"] == "undeployed"
    assert info["deployed_version"] is None
    # undeploying twice is a state error
    assert (await client.post("/api/v1/definitions/echo-agent/undeploy")).status_code == 409
    assert (await client.delete("/api/v1/definitions/echo-agent")).status_code == 204
    assert (await client.get("/api/v1/definitions/echo-agent")).status_code == 404


async def test_export_is_deterministic(client: httpx.AsyncClient) -> None:
    await create_default_resources(client)
    defn = load_example("support-rag.yaml")
    await client.post("/api/v1/definitions", json=defn.canonical_dict())
    first = (await client.get("/api/v1/definitions/support-rag/export")).text
    second = (await client.get("/api/v1/definitions/support-rag/export")).text
    assert first == second


async def test_ephemeral_deploy_serves_draft(client: httpx.AsyncClient) -> None:
    await create_default_resources(client)
    defn = load_example("echo-agent.yaml")
    await client.post("/api/v1/definitions", json=defn.canonical_dict())
    response = await client.post(
        "/api/v1/definitions/echo-agent/deploy", params={"ephemeral": "true"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["endpoint_url"] == "https://api.example/a2a/_draft/echo-agent"
    assert body["registry_id"] is None
    # ephemeral deploys never mark the definition deployed
    info = (await client.get("/api/v1/definitions/echo-agent")).json()
    assert info["status"] == "draft"
    # the draft card is served
    card = await client.get("/a2a/_draft/echo-agent/.well-known/agent-card.json")
    assert card.status_code == 200


@respx.mock
async def test_deploy_registers_with_registry(client: httpx.AsyncClient) -> None:
    entry_id = str(uuid.uuid4())

    def register_responder(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["kind"] == "agent"
        assert payload["url"] == "https://api.example/a2a/echo-agent"
        now = datetime.now(UTC).isoformat()
        return httpx.Response(
            201,
            json={
                "id": entry_id,
                "kind": "agent",
                "card": payload["card"],
                "url": payload["url"],
                "tags": payload.get("tags", []),
                "owner": "anonymous",
                "status": "starting",
                "last_seen": None,
                "created_at": now,
                "updated_at": now,
            },
        )

    register_route = respx.post(f"{REGISTRY_BASE}/api/v1/agents").mock(
        side_effect=register_responder
    )
    delete_route = respx.delete(f"{REGISTRY_BASE}/api/v1/agents/{entry_id}").mock(
        return_value=httpx.Response(204)
    )

    app = create_app(make_settings(registry_url=REGISTRY_BASE))
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://runtime.test"
        ) as local_client:
            await create_default_resources(local_client)
            defn = load_example("echo-agent.yaml")
            await local_client.post("/api/v1/definitions", json=defn.canonical_dict())
            deployed = (await local_client.post("/api/v1/definitions/echo-agent/deploy")).json()
            assert deployed["registry_id"] == entry_id
            assert register_route.called
            assert (
                await local_client.post("/api/v1/definitions/echo-agent/undeploy")
            ).status_code == 204
            assert delete_route.called


async def test_deploy_with_version_label(client: httpx.AsyncClient) -> None:
    """Publishers may label a deploy with a semantic version (FEEDBACK 2.3)."""
    await create_default_resources(client)
    defn = load_example("echo-agent.yaml")
    await client.post("/api/v1/definitions", json=defn.canonical_dict())

    first = await client.post(
        "/api/v1/definitions/echo-agent/deploy", params={"version_label": "1.0.0"}
    )
    assert first.status_code == 200, first.text
    assert first.json()["version"] == 1
    assert first.json()["version_label"] == "1.0.0"

    info = (await client.get("/api/v1/definitions/echo-agent")).json()
    assert info["deployed_version_label"] == "1.0.0"

    # the label is served on the agent card
    card = (await client.get("/a2a/echo-agent/.well-known/agent-card.json")).json()
    assert card["version"] == "1.0.0"

    # a second deploy takes the next counter and its own label
    updated = defn.model_copy(update={"description": "v2"})
    await client.put("/api/v1/definitions/echo-agent", json=updated.canonical_dict())
    second = await client.post(
        "/api/v1/definitions/echo-agent/deploy", params={"version_label": "1.1.0"}
    )
    assert (second.json()["version"], second.json()["version_label"]) == (2, "1.1.0")

    # a known label rolls back to that version instead of freezing a new one
    rollback = await client.post(
        "/api/v1/definitions/echo-agent/deploy", params={"version_label": "1.0.0"}
    )
    assert (rollback.json()["version"], rollback.json()["version_label"]) == (1, "1.0.0")
    info = (await client.get("/api/v1/definitions/echo-agent")).json()
    assert (info["deployed_version"], info["latest_version"]) == (1, 2)


async def test_deploy_rejects_malformed_and_conflicting_version_labels(
    client: httpx.AsyncClient,
) -> None:
    await create_default_resources(client)
    defn = load_example("echo-agent.yaml")
    await client.post("/api/v1/definitions", json=defn.canonical_dict())

    not_semver = await client.post(
        "/api/v1/definitions/echo-agent/deploy", params={"version_label": "v1"}
    )
    assert not_semver.status_code == 422

    await client.post("/api/v1/definitions/echo-agent/deploy", params={"version_label": "1.0.0"})
    both = await client.post(
        "/api/v1/definitions/echo-agent/deploy",
        params={"version": 1, "version_label": "2.0.0"},
    )
    assert both.status_code == 409


@respx.mock
async def test_validate_reports_e022_dimension_mismatch(client: httpx.AsyncClient) -> None:
    """The collection lives on the retrieval node, so E022 points at that node (FEEDBACK 1.3)."""
    respx.get(f"{QDRANT_BASE}/collections/support_docs").mock(
        return_value=httpx.Response(
            200, json={"result": {"config": {"params": {"vectors": {"size": 768}}}}}
        )
    )
    respx.route(host="runtime.test").pass_through()
    await create_default_resources(client)  # resource declares dimension 3
    defn = load_example("support-rag.yaml").canonical_dict()

    body = (await client.post("/api/v1/definitions/validate", json=defn)).json()

    e022 = [i for i in body["issues"] if i["code"] == "E022"]
    assert body["valid"] is False
    assert e022[0]["path"] == "nodes/retrieve_1/config/collection"
    assert "768" in e022[0]["message"]


@respx.mock
async def test_validate_unreachable_vector_db_is_not_e022(client: httpx.AsyncClient) -> None:
    """An unreachable vector DB never yields E022; it surfaces at execution (SPEC §6.3)."""
    respx.get(f"{QDRANT_BASE}/collections/support_docs").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    respx.route(host="runtime.test").pass_through()
    await create_default_resources(client)  # resource declares dimension 3
    defn = load_example("support-rag.yaml").canonical_dict()

    body = (await client.post("/api/v1/definitions/validate", json=defn)).json()

    assert [i for i in body["issues"] if i["code"] == "E022"] == []
    # nothing else is wrong with the flow, so an unreadable dimension leaves it valid
    assert body["valid"] is True
