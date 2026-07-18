# agentplane

A small, protocol-first agent platform. You describe **flows** (agent/tool
pipelines) as declarative YAML definitions, deploy them to the **runtime**,
and each flow is served as an **A2A agent** or an **MCP server** behind a
gateway. The **registry** makes everything discoverable and health-checked.

Every boundary is an open standard — no proprietary protocols:

- **A2A** (Agent-to-Agent, v1.0) for agents
- **MCP** (Model Context Protocol, streamable HTTP) for tools
- **OIDC** for identity (any issuer, e.g. Keycloak)
- **OTLP** (OpenTelemetry) for traces, e.g. into Langfuse
- **OpenAI-compatible API** for LLM access through the gateway

## Packages

| Package | Install | Purpose |
|---|---|---|
| [`agentplane-core`](packages/core) | `pip install agentplane-core` | Definition schema (Pydantic v2), validation with stable `E0xx` codes, shared models |
| [`agentplane-sdk`](packages/sdk) | `pip install agentplane-sdk` | typed async/sync clients + `agentplane` CLI |
| [`agentplane-registry`](packages/registry) | `pip install agentplane-registry` | standalone discovery service: register, tag, search, health |
| [`agentplane-runtime`](packages/runtime) | `pip install agentplane-runtime` | owns definitions & resources, executes flows (LangGraph), serves A2A + MCP endpoints |

Optional extras: `agentplane-registry[semantic]` (semantic search),
`agentplane-registry[postgres]` / `agentplane-runtime[postgres]` (asyncpg).

## A flow definition in 30 seconds

```yaml
schema_version: 1
name: support-rag
description: "Answers support questions from the KB"
expose:
  kind: mcp                      # serve as MCP server ("a2a" for an agent)
  tool_name: search_support_kb
nodes:
  - id: start_1
    type: start
    version: 1
    config:
      input_schema:
        type: object
        properties: { query: { type: string } }
        required: [query]
  - id: retrieve_1
    type: retrieval
    version: 1
    config: { resource: kb-support, collection: support_docs, top_k: 4 }
  - id: call_1
    type: llm_call
    version: 1
    config:
      resource: default-llm
      prompt: "Answer using: {documents}\n\nQuestion: {query}"
  - id: end_1
    type: end
    version: 1
    config: { output_from: call_1.text }
edges:
  - { from: start_1.query, to: retrieve_1.query }
  - { from: start_1.query, to: call_1.query }
  - { from: retrieve_1.documents, to: call_1.documents }
  - { from: call_1.text, to: end_1.input }
```

Node types: `start`, `end`, `llm_call`, `mcp_tool`, `retrieval`, `rerank`
(relevance reranking of retrieved documents), `router` (conditional branches),
`template` (text templating). An `llm_call` with
`history: true` becomes conversational: the runtime feeds the prior turns of
the caller's A2A conversation (`contextId`) to the model — see
[`examples/chat-with-history.yaml`](examples/chat-with-history.yaml).
Credentials never appear in definitions — resources (model providers, vector
DBs, MCP servers) are referenced **by name** and stored encrypted in the
runtime. The JSON Schema of the format lives in
[`schemas/flow-definition.schema.json`](schemas/flow-definition.schema.json)
and is the public contract for builders and AI-generated definitions.

## Using it

```bash
# validate locally (no server needed)
agentplane validate flow.yaml

# deploy against a runtime
export AGENTPLANE_RUNTIME_URL=https://api.example/runtime
export AGENTPLANE_TOKEN=...
agentplane resources create -f resources/default-llm.yaml
agentplane deploy flow.yaml                          # -> https://api.example/mcp/support-rag
agentplane deploy flow.yaml --version-label 1.2.0    # publish under a semantic version

# discover via the registry
export AGENTPLANE_REGISTRY_URL=https://api.example/registry
agentplane search "support" --semantic

# lifecycle without deleting: disabled entries are hidden from discovery and
# not health-checked, but stay listed for their owner
agentplane disable <entry-id>
agentplane enable <entry-id>
```

The registry health-checks every entry (interval configurable), records
status **transitions** per entry (`GET /agents/{id}/history`, retention
configurable) and shows owners by display name. Deleting an entry is
restricted to its owner and admins; team members may edit.

Calling a deployed A2A agent is plain JSON-RPC 2.0 over POST — the binding
requires the `A2A-Version: 1.0` header, and the card at
`{endpoint}/.well-known/agent-card.json` advertises it under
`supportedInterfaces[].protocolBinding`:

```bash
curl -X POST https://api.example/a2a/echo-agent   -H 'A2A-Version: 1.0' -H 'Content-Type: application/json'   -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage",
       "params":{"message":{"messageId":"m1","role":"ROLE_USER",
                            "parts":[{"text":"ping"}]}}}'
```

A2A conversations are persistent when the runtime runs with
`AGENTPLANE_RUNTIME_TASK_STORE=database`: tasks (scoped per endpoint and
caller) survive restarts, and clients restore chat history via the standard
A2A `ListTasks`/`GetTask` methods — the portal chat does exactly that.

Same thing from Python:

```python
from agentplane_sdk import RuntimeClient

async with RuntimeClient("https://api.example/runtime", token="...") as client:
    result = await client.validate(definition)   # ValidationResult with E0xx codes
    info = await client.deploy("support-rag")
    print(info.endpoint_url)
```

The SDK is a thin typed wrapper — everything is plain HTTP
(`/api/v1/definitions`, `/api/v1/resources`, `/api/v1/agents`,
`/api/v1/agents/search`, `/capabilities`), so any language works.

## Running a service standalone

Both services are standalone-capable (SQLite by default, auth optional) and
ship a console script — no uvicorn incantation:

```bash
export AGENTPLANE_RUNTIME_PUBLIC_BASE_URL=http://localhost:8000
export AGENTPLANE_RUNTIME_SECRET_KEY=$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')
export AGENTPLANE_RUNTIME_CORS_ORIGINS=http://localhost:5173   # only without a gateway
agentplane-runtime                                             # or: uvx agentplane-runtime
```

Settings come from the environment (prefix `AGENTPLANE_RUNTIME_`) or a `.env`
file in the working directory. Auth is opt-in and bring-your-own-issuer:
`AUTH_MODE=oidc` plus `OIDC_ISSUER` (any OIDC provider — Keycloak, Auth0,
Entra, …) turns on per-user/per-team enforcement — you see, manage and *call*
only what you own or what belongs to one of your teams (`groups` claim,
configurable via `GROUPS_CLAIM`); admins see everything. With the default
`AUTH_MODE=none` nothing changes. `CORS_ORIGINS` is empty by default: in a
deployed stack agentgateway owns CORS; set it only when a browser talks to the
runtime directly (e.g. a builder playground). Running a registry locally needs
`AGENTPLANE_REGISTRY_ALLOW_PRIVATE_URLS=true`, since gateway URLs on loopback
are otherwise rejected.

## Local platform stack

`deploy/compose/` ships a complete stack: traefik, Keycloak (realm import
with demo users `demo-admin`/`demo-builder`/`demo-user`), Postgres,
agentgateway, registry, runtime, the portal UI (chat, registry management)
and the low-code builder — plus Langfuse behind `--profile langfuse`
(admin-only via a Keycloak gate; traces are chat-focused: one trace per
message with user, session, question, answer and token usage).

```bash
docker compose -f deploy/compose/compose.yaml --profile langfuse up -d --wait
./scripts/smoke.sh    # deploy examples, call them via A2A/MCP, check traces
```

Then open http://app.localhost (portal) or http://builder.localhost. A
production overlay with real domains and ACME TLS is in
`deploy/compose/compose.prod.yaml`.

## Development

```bash
uv sync                                  # one venv, one lockfile
uv run pytest                            # unit tests (fast, no docker)
uv run ruff format . && uv run ruff check --fix .
uv run mypy                              # strict
uv run lint-imports                      # architecture contracts
uv run python -m agentplane_core.schema_export > schemas/flow-definition.schema.json
```

Dependency direction is enforced: `sdk → core`, `registry → core`,
`runtime → core + sdk` — services never import each other and only talk via
their public APIs.

## Releasing

The published version of each package is the `version` in its
`packages/<name>/pyproject.toml`. Bump it there, then run the **Release**
workflow (GitHub → Actions → Release → "Run workflow", or push a `v*` tag).
Versions that already exist on PyPI are skipped, so packages can be released
independently. Publishing uses PyPI Trusted Publishing — no tokens.

## License

MIT
