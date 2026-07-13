"""Round-trip property tests over all example definitions (SPEC §3.1).

``parse(serialize(flow)) == flow`` must hold for every file in examples/.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agentplane_core import FlowDefinition

EXAMPLES_DIR = Path(__file__).parents[3] / "examples"


def example_files() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("*.yaml"))


def load_definition(path: Path) -> FlowDefinition:
    with path.open(encoding="utf-8") as fh:
        return FlowDefinition.model_validate(yaml.safe_load(fh))


@pytest.mark.parametrize("path", example_files(), ids=lambda p: p.name)
def test_parse_serialize_roundtrip(path: Path) -> None:
    definition = load_definition(path)
    dumped = definition.canonical_dict()
    reparsed = FlowDefinition.model_validate(dumped)
    assert reparsed == definition
    # serializing again yields byte-identical output (determinism)
    assert json.dumps(reparsed.canonical_dict(), sort_keys=False) == json.dumps(
        dumped, sort_keys=False
    )


@pytest.mark.parametrize("path", example_files(), ids=lambda p: p.name)
def test_nodes_and_edges_are_sorted(path: Path) -> None:
    definition = load_definition(path)
    node_ids = [node.id for node in definition.nodes]
    assert node_ids == sorted(node_ids)
    edge_keys = [(edge.from_, edge.to) for edge in definition.edges]
    assert edge_keys == sorted(edge_keys)


def test_input_order_does_not_change_serialization(echo_definition: FlowDefinition) -> None:
    dumped = echo_definition.canonical_dict()
    nodes = dumped["nodes"]
    edges = dumped["edges"]
    assert isinstance(nodes, list) and isinstance(edges, list)
    shuffled = {**dumped, "nodes": list(reversed(nodes)), "edges": list(reversed(edges))}
    reparsed = FlowDefinition.model_validate(shuffled)
    assert reparsed == echo_definition
    assert reparsed.canonical_dict() == dumped


def test_edge_serializes_with_from_alias(echo_definition: FlowDefinition) -> None:
    dumped = echo_definition.canonical_dict()
    edges = dumped["edges"]
    assert isinstance(edges, list)
    first = edges[0]
    assert isinstance(first, dict)
    assert set(first) == {"from", "to"}
