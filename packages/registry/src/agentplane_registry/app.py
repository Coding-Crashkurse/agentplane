"""Registry app factory: standalone-capable, SQLite by default, auth optional."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agentplane_registry.api import RegistryState, health_router, router
from agentplane_registry.auth import Authenticator
from agentplane_registry.db import Database
from agentplane_registry.embeddings import EmbeddingsClient
from agentplane_registry.health import HealthJob
from agentplane_registry.search import RegistrySearch
from agentplane_registry.settings import REGISTRY_VERSION, RegistrySettings
from agentplane_registry.tracing import setup_tracing


def create_app(settings: RegistrySettings | None = None, *, run_health_job: bool = True) -> FastAPI:
    """Build the registry service; everything optional degrades gracefully."""
    cfg = settings or RegistrySettings()
    db = Database(cfg.db_url)
    embeddings = (
        EmbeddingsClient(cfg.embeddings_base_url, cfg.embeddings_model)
        if cfg.embeddings_base_url and cfg.embeddings_model
        else None
    )
    search = RegistrySearch(db, embeddings)
    health_job = HealthJob(db, cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await db.create_all()
        if run_health_job:
            health_job.start()
        try:
            yield
        finally:
            if run_health_job:
                await health_job.stop()
            await db.dispose()

    app = FastAPI(title="agentplane-registry", version=REGISTRY_VERSION, lifespan=lifespan)
    app.state.registry = RegistryState(db=db, settings=cfg, search=search)
    app.state.authenticator = Authenticator(cfg)
    app.state.health_job = health_job
    app.include_router(router)
    app.include_router(health_router)
    setup_tracing(app, service_name="agentplane-registry")
    return app


def main() -> None:  # pragma: no cover - process entrypoint
    import uvicorn  # noqa: PLC0415 - optional server dep, only needed for the entrypoint

    uvicorn.run(create_app(), host="0.0.0.0", port=8100)


__all__ = ["create_app", "main"]
