"""A2A task persistence (SPEC §6.5): TASK_STORE=database survives restarts."""

from __future__ import annotations

import contextlib
from pathlib import Path

import httpx
import pytest
import respx
from cryptography.fernet import Fernet

from agentplane_core import ModelProviderResource
from agentplane_runtime.db import Database
from agentplane_runtime.resources import ResourceConflictError, ResourceService
from agentplane_runtime.secrets import FernetSecretsProvider
from agentplane_runtime.serving import EndpointManager

from .conftest import LLM_BASE, SSE_PONG, load_example, make_settings


async def _database(tmp_path: Path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}")
    await db.create_all()
    return db


async def _manager(db: Database) -> EndpointManager:
    secrets = FernetSecretsProvider(db, Fernet.generate_key().decode("ascii"))
    resources = ResourceService(db, secrets)
    # conflict = second manager on the same database (simulated restart)
    with contextlib.suppress(ResourceConflictError):
        await resources.create(
            ModelProviderResource(
                name="default-llm", base_url=LLM_BASE, default_model="gpt-5-mini"
            ),
            "anonymous",
        )
    settings = make_settings(task_store="database")
    return EndpointManager(resources, settings, engine=db.engine)


def _rpc(method: str, params: dict[str, object], request_id: int) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


def _send_params(text: str, message_id: str) -> dict[str, object]:
    return {"message": {"messageId": message_id, "role": "ROLE_USER", "parts": [{"text": text}]}}


@respx.mock
async def test_tasks_persist_across_endpoint_restarts(tmp_path: Path) -> None:
    respx.post(f"{LLM_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200, text=SSE_PONG, headers={"content-type": "text/event-stream"}
        )
    )
    respx.route(host="rt.test").pass_through()
    db = await _database(tmp_path)
    defn = load_example("echo-agent.yaml")

    manager = await _manager(db)
    await manager.start(defn, 1)
    transport = httpx.ASGITransport(app=manager.a2a)
    async with httpx.AsyncClient(transport=transport, base_url="http://rt.test") as client:
        headers = {"A2A-Version": "1.0"}
        first = await client.post(
            "/echo-agent/", json=_rpc("SendMessage", _send_params("ping", "m1"), 1), headers=headers
        )
        task = first.json()["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"
        context_id = task["contextId"]

        listed = await client.post(
            "/echo-agent/",
            json=_rpc(
                "ListTasks",
                {"pageSize": 10, "includeArtifacts": True, "historyLength": 50},
                2,
            ),
            headers=headers,
        )
        tasks = listed.json()["result"]["tasks"]
        assert [t["contextId"] for t in tasks] == [context_id]
        history_texts = [
            part["text"] for message in tasks[0]["history"] for part in message["parts"]
        ]
        assert "ping" in history_texts
    await manager.stop_all()

    # A fresh manager on the same database (simulated restart) still sees the task.
    manager = await _manager(db)
    await manager.start(defn, 1)
    transport = httpx.ASGITransport(app=manager.a2a)
    async with httpx.AsyncClient(transport=transport, base_url="http://rt.test") as client:
        listed = await client.post(
            "/echo-agent/",
            json=_rpc("ListTasks", {"pageSize": 10}, 3),
            headers={"A2A-Version": "1.0"},
        )
        tasks = listed.json()["result"]["tasks"]
        assert len(tasks) == 1
        assert tasks[0]["contextId"] == context_id
    await manager.stop_all()
    await db.dispose()


@pytest.mark.parametrize("ephemeral", [True])
@respx.mock
async def test_draft_endpoints_stay_in_memory(tmp_path: Path, ephemeral: bool) -> None:
    respx.post(f"{LLM_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200, text=SSE_PONG, headers={"content-type": "text/event-stream"}
        )
    )
    respx.route(host="rt.test").pass_through()
    db = await _database(tmp_path)
    defn = load_example("echo-agent.yaml")
    manager = await _manager(db)
    await manager.start(defn, 1, ephemeral=ephemeral)
    transport = httpx.ASGITransport(app=manager.a2a)
    async with httpx.AsyncClient(transport=transport, base_url="http://rt.test") as client:
        headers = {"A2A-Version": "1.0"}
        sent = await client.post(
            "/_draft/echo-agent/",
            json=_rpc("SendMessage", _send_params("ping", "m1"), 1),
            headers=headers,
        )
        assert sent.json()["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
    await manager.stop_all()

    # Restart: the draft's task must NOT have been persisted.
    manager = await _manager(db)
    await manager.start(defn, 1)
    transport = httpx.ASGITransport(app=manager.a2a)
    async with httpx.AsyncClient(transport=transport, base_url="http://rt.test") as client:
        listed = await client.post(
            "/echo-agent/",
            json=_rpc("ListTasks", {"pageSize": 10}, 2),
            headers={"A2A-Version": "1.0"},
        )
        assert listed.json()["result"].get("tasks", []) == []
    await manager.stop_all()
    await db.dispose()
