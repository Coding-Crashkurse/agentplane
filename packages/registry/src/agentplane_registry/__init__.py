"""agentplane-registry: standalone-capable discovery service."""

from agentplane_registry.app import create_app
from agentplane_registry.settings import REGISTRY_VERSION, RegistrySettings

__version__ = REGISTRY_VERSION

__all__ = ["REGISTRY_VERSION", "RegistrySettings", "create_app"]
