"""CLI behavior and exit codes (SPEC §4.2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import respx
import yaml
from httpx import Response
from typer.testing import CliRunner

from agentplane_sdk.cli import app
from agentplane_sdk.config import resolve_config

runner = CliRunner()
RUNTIME = "http://runtime.test"


def test_validate_ok(echo_yaml_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(echo_yaml_path)])
    assert result.exit_code == 0


def test_validate_invalid_definition(tmp_path: Path, echo_yaml_path: Path) -> None:
    data = yaml.safe_load(echo_yaml_path.read_text(encoding="utf-8"))
    data["schema_version"] = 99
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(data), encoding="utf-8")
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1
    assert "E001" in result.output


def test_validate_json_output(echo_yaml_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(echo_yaml_path), "--json"])
    assert result.exit_code == 0
    assert '"valid": true' in result.output


def test_missing_file_exits_3() -> None:
    result = runner.invoke(app, ["validate", "does-not-exist.yaml"])
    assert result.exit_code == 3


@respx.mock
def test_deploy_creates_then_deploys(echo_yaml_path: Path) -> None:
    respx.post(f"{RUNTIME}/api/v1/definitions").mock(
        return_value=Response(201, json=_definition_info())
    )
    respx.post(f"{RUNTIME}/api/v1/definitions/echo-agent/deploy").mock(
        return_value=Response(
            200,
            json={
                "name": "echo-agent",
                "version": 1,
                "endpoint_url": "https://gw/a2a/echo-agent",
                "registry_id": None,
            },
        )
    )
    result = runner.invoke(app, ["deploy", str(echo_yaml_path), "--runtime-url", RUNTIME])
    assert result.exit_code == 0, result.output
    assert "deployed echo-agent v1" in result.output


@respx.mock
def test_deploy_falls_back_to_update_on_conflict(echo_yaml_path: Path) -> None:
    respx.post(f"{RUNTIME}/api/v1/definitions").mock(return_value=Response(409, text="exists"))
    respx.put(f"{RUNTIME}/api/v1/definitions/echo-agent").mock(
        return_value=Response(200, json=_definition_info())
    )
    respx.post(f"{RUNTIME}/api/v1/definitions/echo-agent/deploy").mock(
        return_value=Response(
            200,
            json={
                "name": "echo-agent",
                "version": 2,
                "endpoint_url": "https://gw/a2a/echo-agent",
                "registry_id": None,
            },
        )
    )
    result = runner.invoke(app, ["deploy", str(echo_yaml_path), "--runtime-url", RUNTIME])
    assert result.exit_code == 0, result.output
    assert "v2" in result.output


def test_deploy_without_runtime_url_exits_2(
    echo_yaml_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mp = monkeypatch
    mp.delenv("AGENTPLANE_RUNTIME_URL", raising=False)
    mp.setattr("agentplane_sdk.config.CONFIG_PATH", Path("Z:/nonexistent/config.toml"))
    result = runner.invoke(app, ["deploy", str(echo_yaml_path)])
    assert result.exit_code == 2


def test_config_resolution_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mp = monkeypatch
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        'runtime_url = "http://from-file"\nregistry_url = "http://reg-file"\n', encoding="utf-8"
    )
    mp.setenv("AGENTPLANE_RUNTIME_URL", "http://from-env")
    config = resolve_config(config_path=config_file)
    assert config.runtime_url == "http://from-env"  # env beats file
    assert config.registry_url == "http://reg-file"  # file fills the gap
    config = resolve_config(runtime_url="http://from-flag", config_path=config_file)
    assert config.runtime_url == "http://from-flag"  # flag beats env


def _definition_info() -> dict[str, object]:
    return {
        "name": "echo-agent",
        "display_name": "Echo Agent",
        "expose_kind": "a2a",
        "status": "draft",
        "created_at": "2026-07-12T00:00:00Z",
        "updated_at": "2026-07-12T00:00:00Z",
    }
