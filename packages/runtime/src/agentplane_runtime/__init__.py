"""agentplane-runtime: definitions, resources, flow execution, A2A + MCP serving."""

from agentplane_runtime.app import create_app
from agentplane_runtime.settings import RUNTIME_VERSION, RuntimeSettings

__version__ = RUNTIME_VERSION

__all__ = ["RUNTIME_VERSION", "RuntimeSettings", "create_app"]
