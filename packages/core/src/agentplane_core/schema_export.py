"""Export the FlowDefinition JSON Schema (SPEC §3.6).

Usage::

    uv run python -m agentplane_core.schema_export > schemas/flow-definition.schema.json

The output is the public contract for the builder and AI-generated
definitions; it is committed and diffed in CI.
"""

from __future__ import annotations

import json

from agentplane_core.definition import FlowDefinition


def export_schema_json() -> str:
    """Render the FlowDefinition JSON Schema deterministically."""
    schema = FlowDefinition.model_json_schema(by_alias=True)
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://agentplane.dev/schemas/flow-definition.schema.json"
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> None:
    print(export_schema_json(), end="")


if __name__ == "__main__":
    main()
