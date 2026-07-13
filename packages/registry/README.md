# agentplane-registry

Standalone-capable discovery service for A2A agents and MCP servers:
register, tag, search (text + optional semantic), health-check.

- Runs alone: SQLite by default, auth optional (generic OIDC, any issuer),
  no calls to other agentplane services.
- Extras: `agentplane-registry[semantic]` (numpy + embeddings via an
  OpenAI-compatible endpoint), `[postgres]` (asyncpg). Without extras the
  service degrades gracefully and announces features via `GET /capabilities`.
- Stores **gateway URLs only**; private/internal hosts are rejected unless
  `AGENTPLANE_REGISTRY_ALLOW_PRIVATE_URLS=true`.

Run it:

```
uvicorn --factory agentplane_registry.app:create_app --port 8100
```
