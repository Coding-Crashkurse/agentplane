from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agentplane_core import FlowDefinition

EXAMPLES_DIR = Path(__file__).parents[3] / "examples"


@pytest.fixture
def echo_definition() -> FlowDefinition:
    with (EXAMPLES_DIR / "echo-agent.yaml").open(encoding="utf-8") as fh:
        return FlowDefinition.model_validate(yaml.safe_load(fh))
