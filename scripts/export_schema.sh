#!/usr/bin/env bash
# Regenerate the committed FlowDefinition JSON Schema (SPEC §3.6).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m agentplane_core.schema_export > schemas/flow-definition.schema.json
echo "wrote schemas/flow-definition.schema.json"
