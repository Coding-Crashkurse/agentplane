"""Runtime app factory: API + dynamic /a2a and /mcp endpoint serving."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentplane_runtime.api import RuntimeState, health_router, router
from agentplane_runtime.auth import Authenticator
from agentplane_runtime.db import Database
from agentplane_runtime.definitions import DefinitionService
from agentplane_runtime.migrate import run_migrations
from agentplane_runtime.registration import RegistryRegistrar
from agentplane_runtime.resources import ResourceService
from agentplane_runtime.secrets import FernetSecretsProvider
from agentplane_runtime.serving import EndpointManager
from agentplane_runtime.settings import RUNTIME_VERSION, RuntimeSettings
from agentplane_runtime.tracing import setup_tracing

logger = logging.getLogger(__name__)

_REAP_INTERVAL_S = 60.0


def create_app(settings: RuntimeSettings | None = None) -> FastAPI:
    cfg = settings or RuntimeSettings()
    if not cfg.public_base_url:
        raise RuntimeError("AGENTPLANE_RUNTIME_PUBLIC_BASE_URL is required")
    if not cfg.secret_key:
        raise RuntimeError("AGENTPLANE_RUNTIME_SECRET_KEY is required")

    db = Database(cfg.db_url)
    secrets = FernetSecretsProvider(db, cfg.secret_key)
    resources = ResourceService(db, secrets)
    authenticator = Authenticator(cfg)
    endpoints = EndpointManager(resources, cfg, authenticator)
    registrar = RegistryRegistrar(cfg)
    definitions = DefinitionService(db, resources, endpoints, registrar)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await run_migrations(db.engine)
        try:
            await definitions.restore_deployed_endpoints()
        except Exception:
            logger.exception("failed to restore deployed endpoints")

        async def reaper() -> None:
            while True:
                await asyncio.sleep(_REAP_INTERVAL_S)
                await endpoints.reap_expired()

        reap_task = asyncio.create_task(reaper(), name="ephemeral-reaper")
        try:
            yield
        finally:
            reap_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reap_task
            await registrar.shutdown()
            await endpoints.stop_all()
            await db.dispose()

    app = FastAPI(title="agentplane-runtime", version=RUNTIME_VERSION, lifespan=lifespan)
    app.state.runtime = RuntimeState(
        definitions=definitions,
        resources=resources,
        auth_mode=cfg.auth_mode,
        builder_role=cfg.builder_role,
    )
    app.state.authenticator = authenticator
    app.state.endpoints = endpoints
    app.include_router(router)
    app.include_router(health_router)
    app.mount("/a2a", endpoints.a2a)
    app.mount("/mcp", endpoints.mcp)
    if cfg.cors_origins:
        # Outermost middleware, so it also covers the mounted /a2a and /mcp
        # endpoints — that is the point: a browser talking A2A without a gateway.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    setup_tracing(app, service_name="agentplane-runtime")
    return app


def main() -> None:  # pragma: no cover - process entrypoint
    import uvicorn  # noqa: PLC0415 - only needed for the entrypoint

    cfg = RuntimeSettings()
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port)


__all__ = ["create_app", "main"]
