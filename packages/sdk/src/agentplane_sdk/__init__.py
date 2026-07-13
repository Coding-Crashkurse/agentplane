"""agentplane-sdk: thin typed client + CLI."""

from agentplane_sdk.auth import (
    OidcClientCredentialsProvider,
    StaticTokenProvider,
    TokenProvider,
    as_token_provider,
)
from agentplane_sdk.client import (
    RegistryClient,
    RuntimeClient,
    SyncRegistryClient,
    SyncRuntimeClient,
)
from agentplane_sdk.errors import (
    AgentplaneError,
    ApiError,
    AuthError,
    ConflictError,
    NotFoundError,
    TransportError,
    ValidationFailedError,
)

__version__ = "0.0.1"

__all__ = [
    "AgentplaneError",
    "ApiError",
    "AuthError",
    "ConflictError",
    "NotFoundError",
    "OidcClientCredentialsProvider",
    "RegistryClient",
    "RuntimeClient",
    "StaticTokenProvider",
    "SyncRegistryClient",
    "SyncRuntimeClient",
    "TokenProvider",
    "TransportError",
    "ValidationFailedError",
    "as_token_provider",
]
