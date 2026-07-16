"""OIDC ownership enforcement on the runtime API (SPEC §7.1).

With ``auth_mode=oidc`` a user sees and manages only their own definitions and
resources; an admin is unrestricted. In ``auth_mode=none`` everything is a single
``anonymous`` owner (covered by the other API tests) — enforcement is a no-op.
Writers additionally carry the ``builder`` role (write gating has its own tests
in ``test_builder_role.py``); ownership visibility is what is under test here.
"""

from __future__ import annotations

import httpx
import respx

from .conftest import auth_header, llm_resource_body, load_example, make_token, mock_issuer


async def _seed_alice_flow(client: httpx.AsyncClient, alice: str) -> None:
    assert (
        await client.post("/api/v1/resources", json=llm_resource_body(), headers=auth_header(alice))
    ).status_code == 201
    defn = load_example("echo-agent.yaml").canonical_dict()
    assert (
        await client.post("/api/v1/definitions", json=defn, headers=auth_header(alice))
    ).status_code == 201


@respx.mock
async def test_request_without_token_is_401(oidc_client: httpx.AsyncClient) -> None:
    assert (await oidc_client.get("/api/v1/definitions")).status_code == 401


@respx.mock
async def test_user_sees_only_own_definitions(oidc_client: httpx.AsyncClient) -> None:
    mock_issuer()
    alice, bob = make_token("alice", ["user", "builder"]), make_token("bob", ["user"])
    await _seed_alice_flow(oidc_client, alice)

    alice_list = (await oidc_client.get("/api/v1/definitions", headers=auth_header(alice))).json()
    assert [d["name"] for d in alice_list] == ["echo-agent"]

    bob_list = (await oidc_client.get("/api/v1/definitions", headers=auth_header(bob))).json()
    assert bob_list == []
    # invisible, not forbidden — no existence leak
    assert (
        await oidc_client.get("/api/v1/definitions/echo-agent", headers=auth_header(bob))
    ).status_code == 404


@respx.mock
async def test_user_cannot_mutate_anothers_definition(oidc_client: httpx.AsyncClient) -> None:
    mock_issuer()
    alice = make_token("alice", ["user", "builder"])
    bob = make_token("bob", ["user", "builder"])
    await _seed_alice_flow(oidc_client, alice)
    defn = load_example("echo-agent.yaml").canonical_dict()

    assert (
        await oidc_client.put("/api/v1/definitions/echo-agent", json=defn, headers=auth_header(bob))
    ).status_code == 404
    assert (
        await oidc_client.post("/api/v1/definitions/echo-agent/deploy", headers=auth_header(bob))
    ).status_code == 404
    assert (
        await oidc_client.get("/api/v1/definitions/echo-agent/export", headers=auth_header(bob))
    ).status_code == 404
    assert (
        await oidc_client.delete("/api/v1/definitions/echo-agent", headers=auth_header(bob))
    ).status_code == 404
    # owner is untouched
    assert (
        await oidc_client.get("/api/v1/definitions/echo-agent", headers=auth_header(alice))
    ).status_code == 200


@respx.mock
async def test_admin_sees_everyones_definitions(oidc_client: httpx.AsyncClient) -> None:
    mock_issuer()
    alice = make_token("alice", ["user", "builder"])
    admin = make_token("root", ["admin", "user"])
    await _seed_alice_flow(oidc_client, alice)
    admin_list = (await oidc_client.get("/api/v1/definitions", headers=auth_header(admin))).json()
    assert [d["name"] for d in admin_list] == ["echo-agent"]
    assert (
        await oidc_client.get("/api/v1/definitions/echo-agent", headers=auth_header(admin))
    ).status_code == 200


@respx.mock
async def test_user_sees_only_own_resources(oidc_client: httpx.AsyncClient) -> None:
    mock_issuer()
    alice = make_token("alice", ["user", "builder"])
    bob = make_token("bob", ["user", "builder"])
    assert (
        await oidc_client.post(
            "/api/v1/resources", json=llm_resource_body(), headers=auth_header(alice)
        )
    ).status_code == 201

    alice_res = (await oidc_client.get("/api/v1/resources", headers=auth_header(alice))).json()
    assert [r["name"] for r in alice_res] == ["default-llm"]
    assert (await oidc_client.get("/api/v1/resources", headers=auth_header(bob))).json() == []
    assert (
        await oidc_client.get("/api/v1/resources/default-llm", headers=auth_header(bob))
    ).status_code == 404
    assert (
        await oidc_client.delete("/api/v1/resources/default-llm", headers=auth_header(bob))
    ).status_code == 404


@respx.mock
async def test_teammates_share_group_flows_and_resources(oidc_client: httpx.AsyncClient) -> None:
    """user_x publishes into a team; a teammate can see AND edit; a stranger cannot."""
    mock_issuer()
    user_x = make_token("user-x", ["user", "builder"], groups=["team-payments"])
    teammate = make_token("user-y", ["user", "builder"], groups=["team-payments"])
    stranger = make_token("user-z", ["user", "builder"], groups=["team-other"])

    assert (
        await oidc_client.post(
            "/api/v1/resources",
            json=llm_resource_body(),
            params={"group": "team-payments"},
            headers=auth_header(user_x),
        )
    ).status_code == 201
    defn = load_example("echo-agent.yaml").canonical_dict()
    created = await oidc_client.post(
        "/api/v1/definitions",
        json=defn,
        params={"group": "team-payments"},
        headers=auth_header(user_x),
    )
    assert created.status_code == 201
    assert (created.json()["owner"], created.json()["group"]) == ("user-x", "team-payments")

    # teammate: sees the flow and the resource, can read and edit
    teammate_list = (
        await oidc_client.get("/api/v1/definitions", headers=auth_header(teammate))
    ).json()
    assert [d["name"] for d in teammate_list] == ["echo-agent"]
    teammate_res = (
        await oidc_client.get("/api/v1/resources", headers=auth_header(teammate))
    ).json()
    assert [r["name"] for r in teammate_res] == ["default-llm"]
    assert (
        await oidc_client.put(
            "/api/v1/definitions/echo-agent", json=defn, headers=auth_header(teammate)
        )
    ).status_code == 200

    # stranger: sees nothing, mutates nothing
    assert (
        await oidc_client.get("/api/v1/definitions", headers=auth_header(stranger))
    ).json() == []
    assert (
        await oidc_client.get("/api/v1/definitions/echo-agent", headers=auth_header(stranger))
    ).status_code == 404
    assert (
        await oidc_client.put(
            "/api/v1/definitions/echo-agent", json=defn, headers=auth_header(stranger)
        )
    ).status_code == 404
    assert (await oidc_client.get("/api/v1/resources", headers=auth_header(stranger))).json() == []


@respx.mock
async def test_cannot_assign_a_foreign_group(oidc_client: httpx.AsyncClient) -> None:
    mock_issuer()
    user_x = make_token("user-x", ["user", "builder"], groups=["team-payments"])
    assert (
        await oidc_client.post(
            "/api/v1/resources",
            json=llm_resource_body(),
            params={"group": "team-secret"},
            headers=auth_header(user_x),
        )
    ).status_code == 403


def _rpc_send(text: str) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "SendMessage",
        "params": {"message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": text}]}},
    }


@respx.mock
async def test_invocation_is_gated_by_the_same_team_predicate(
    oidc_client: httpx.AsyncClient,
) -> None:
    """Calling a served endpoint follows owner-or-team, discovery stays public (Phase 3).

    Invocation is not a write: the teammate carries only the ``user`` role and
    still calls the endpoint — the builder role gates the definitions API only.
    """
    mock_issuer()
    user_x = make_token("user-x", ["user", "builder"], groups=["team-payments"])
    teammate = make_token("user-y", ["user"], groups=["team-payments"])
    stranger = make_token("user-z", ["user"], groups=["team-other"])

    assert (
        await oidc_client.post(
            "/api/v1/resources",
            json=llm_resource_body(),
            params={"group": "team-payments"},
            headers=auth_header(user_x),
        )
    ).status_code == 201
    defn = load_example("echo-agent.yaml").canonical_dict()
    assert (
        await oidc_client.post(
            "/api/v1/definitions",
            json=defn,
            params={"group": "team-payments"},
            headers=auth_header(user_x),
        )
    ).status_code == 201
    deployed = await oidc_client.post(
        "/api/v1/definitions/echo-agent/deploy", headers=auth_header(user_x)
    )
    assert deployed.status_code == 200, deployed.text

    # discovery stays public: the agent card carries no secrets and the
    # registry health job fetches it unauthenticated
    card = await oidc_client.get("/a2a/echo-agent/.well-known/agent-card.json")
    assert card.status_code == 200

    rpc_headers = {"A2A-Version": "1.0"}
    no_token = await oidc_client.post("/a2a/echo-agent/", json=_rpc_send("hi"), headers=rpc_headers)
    assert no_token.status_code == 401

    as_stranger = await oidc_client.post(
        "/a2a/echo-agent/", json=_rpc_send("hi"), headers={**rpc_headers, **auth_header(stranger)}
    )
    assert as_stranger.status_code == 403

    sse = 'data: {"choices":[{"delta":{"content":"pong"}}]}\n\ndata: [DONE]\n\n'
    respx.post("http://llm.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
    )
    as_teammate = await oidc_client.post(
        "/a2a/echo-agent/", json=_rpc_send("ping"), headers={**rpc_headers, **auth_header(teammate)}
    )
    assert as_teammate.status_code == 200, as_teammate.text
    task = as_teammate.json()["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["parts"][0]["text"] == "pong"
