"""Gateway-URL guard: reject private/internal hosts (SPEC §3.4).

The registry stores gateway URLs only. Unless ``ALLOW_PRIVATE_URLS`` is set,
URLs resolving to loopback/private/link-local hosts are refused so internal
service addresses can never leak into the registry.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

_PRIVATE_HOSTNAMES = {"localhost", "host.docker.internal"}
_PRIVATE_SUFFIXES = (".local", ".internal", ".localdomain")


def is_private_url(url: str) -> bool:
    """True when the URL's host looks private/internal (best-effort, no DNS)."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return True
    host = parsed.hostname.lower()
    if host in _PRIVATE_HOSTNAMES or host.endswith(_PRIVATE_SUFFIXES):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return "." not in host  # bare docker-compose service names etc.
    return address.is_private or address.is_loopback or address.is_link_local


__all__ = ["is_private_url"]
