from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agentplane_core import FlowDefinition

REPO_ROOT = Path(__file__).parents[3]


@pytest.fixture
def echo_definition() -> FlowDefinition:
    with (REPO_ROOT / "examples" / "echo-agent.yaml").open(encoding="utf-8") as fh:
        return FlowDefinition.model_validate(yaml.safe_load(fh))


@pytest.fixture
def echo_yaml_path() -> Path:
    return REPO_ROOT / "examples" / "echo-agent.yaml"
