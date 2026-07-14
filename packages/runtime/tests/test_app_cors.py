"""Optional CORS middleware (FEEDBACK 1.2).

Without a gateway in front of the runtime, a browser cannot talk A2A to
``/a2a/{name}`` at all — the builder playground chats straight from the page.
In production agentgateway owns CORS, so the middleware stays off by default.
"""

from __future__ import annotations

import httpx
from asgi_lifespan import LifespanManager

from agentplane_runtime.app import create_app

from .conftest import make_settings

ORIGIN = "http://builder.localhost:5173"


async def _preflight(**settings: object) -> httpx.Response:
    app = create_app(make_settings(**settings))
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://runtime.test") as client:
            return await client.options(
                "/a2a/echo-agent/",
                headers={
                    "Origin": ORIGIN,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type,a2a-version",
                },
            )


async def test_no_cors_headers_by_default() -> None:
    response = await _preflight()
    assert "access-control-allow-origin" not in response.headers


async def test_configured_origin_is_allowed_on_the_a2a_mount() -> None:
    response = await _preflight(cors_origins=[ORIGIN])
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert "POST" in response.headers["access-control-allow-methods"]


async def test_origins_come_from_a_comma_separated_env_value() -> None:
    settings = make_settings(cors_origins="http://a.test, http://b.test")
    assert settings.cors_origins == ["http://a.test", "http://b.test"]
