"""SDK error hierarchy mapped from HTTP responses."""

from __future__ import annotations

from agentplane_core import ValidationResult


class AgentplaneError(Exception):
    """Base class for all SDK errors."""


class TransportError(AgentplaneError):
    """Network-level failure (connect, timeout, DNS)."""


class AuthError(AgentplaneError):
    """401/403 or token acquisition failure."""


class NotFoundError(AgentplaneError):
    """404 for a named object or entry."""


class ConflictError(AgentplaneError):
    """409 — duplicate name or referenced object."""


class ApiError(AgentplaneError):
    """Any other non-2xx response."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class ValidationFailedError(AgentplaneError):
    """422 with a machine-readable ValidationResult."""

    def __init__(self, result: ValidationResult) -> None:
        lines = ", ".join(f"{i.code}@{i.path}" for i in result.issues)
        super().__init__(f"validation failed: {lines}")
        self.result = result


__all__ = [
    "AgentplaneError",
    "ApiError",
    "AuthError",
    "ConflictError",
    "NotFoundError",
    "TransportError",
    "ValidationFailedError",
]
