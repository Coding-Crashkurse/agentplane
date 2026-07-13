# agentplane-runtime

Owns flow definitions and resources; executes flows (LangGraph); serves each
deployed flow as an **A2A agent** (`/a2a/{name}`, a2a-sdk, A2A v1.0) or an
**MCP server** (`/mcp/{name}`, FastMCP streamable HTTP); self-registers with
the agentplane registry.

Required configuration (env prefix `AGENTPLANE_RUNTIME_`):

| Variable | Purpose |
|---|---|
| `PUBLIC_BASE_URL` | externally reachable base (a gateway route) |
| `REGISTRY_URL` | where to self-register (empty disables registration) |
| `SECRET_KEY` | Fernet key for resource credentials |
| `LLM_BASE_URL` | gateway's OpenAI-compatible endpoint (resource default) |

Run it:

```
uvicorn --factory agentplane_runtime.app:create_app --port 8000
```
