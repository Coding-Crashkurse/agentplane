"""Process entrypoint: ``agentplane-runtime`` (console script) / ``python -m agentplane_runtime``.

Configuration comes from the environment (prefix ``AGENTPLANE_RUNTIME_``) or a
``.env`` file in the working directory — no uvicorn incantation needed.
"""

from __future__ import annotations

from agentplane_runtime.app import main

if __name__ == "__main__":  # pragma: no cover - process entrypoint
    main()

__all__ = ["main"]
