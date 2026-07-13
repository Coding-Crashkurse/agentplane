"""The exported JSON Schema is the public contract (SPEC §3.6)."""

from __future__ import annotations

import json
from pathlib import Path

from agentplane_core.schema_export import export_schema_json

REPO_ROOT = Path(__file__).parents[3]


def test_export_is_deterministic_json() -> None:
    first = export_schema_json()
    second = export_schema_json()
    assert first == second
    schema = json.loads(first)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "FlowDefinition" in schema.get("title", "") or "properties" in schema


def test_schema_covers_node_types() -> None:
    schema = json.loads(export_schema_json())
    text = json.dumps(schema)
    for node_type in ("start", "end", "llm_call", "mcp_tool", "retrieval"):
        assert node_type in text


def test_committed_schema_is_current() -> None:
    """schemas/ is generated + committed; a stale file must fail CI."""
    committed = REPO_ROOT / "schemas" / "flow-definition.schema.json"
    assert committed.exists(), "run: uv run python -m agentplane_core.schema_export"
    assert committed.read_text(encoding="utf-8") == export_schema_json()
