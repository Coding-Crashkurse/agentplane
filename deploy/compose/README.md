# agentplane compose stack

The full SaaS-shaped local stack: traefik, Keycloak, postgres, agentgateway,
registry, runtime, portal UI, builder, and a switchable tracing backend
(Langfuse v3 or MLflow) behind a shared OTel collector.

All client-visible URLs go through traefik + agentgateway; the registry and
runtime are never exposed directly. The traefik container carries the public
hostnames (`auth.localhost`, `api.localhost`, ...) as network aliases, so a
token minted via `http://auth.localhost` carries the same issuer whether it is
validated from your browser or from inside a service container.

## Bootstrap

```bash
docker compose -f deploy/compose/compose.yaml --profile langfuse up -d --wait
```

That is the standard invocation. The first start pulls/builds images and runs
DB migrations; `--wait` blocks until healthchecks pass. The `ui` and `builder`
services fall back to building from the sibling checkouts `../agentplane-ui`
and `../langgraph_a2a` until CI-published images exist.

| URL | Service | Login |
|---|---|---|
| http://app.localhost | portal UI | see demo users below |
| http://builder.localhost | agent builder | demo-admin or demo-builder |
| http://api.localhost | agentgateway (A2A/MCP/registry/runtime APIs) | JWT |
| http://auth.localhost | Keycloak | `admin` / `admin` (bootstrap) |
| http://langfuse.localhost | Langfuse (profile `langfuse`) | `admin@example.com` / `admin12345` |
| http://mlflow.localhost | MLflow (profile `mlflow`) | none |

Demo users (realm `agentplane`):

| User | Password | Roles |
|---|---|---|
| `demo-admin` | `admin` | `admin`, `user`, `builder`, member of `platform-admins` |
| `demo-builder` | `builder` | `user`, `builder` |
| `demo-user` | `user` | `user` |

Self-registration and password reset are enabled on the realm. Email
verification is off because the local stack has no SMTP server — in
production configure SMTP in the realm and turn `verifyEmail` on.

Note: `deploy/compose/postgres-init/` only runs on a fresh `pgdata` volume. If
you upgraded from an older stack, create the new `mlflow` database manually
(`docker compose exec postgres createdb -U agentplane mlflow`) or reset the
volume.

## Superuser first steps (fine-grained admin permissions)

`platform-admins` members get the `admin` realm role plus the
`view-users`/`query-users`/`manage-users` client roles of `realm-management`,
so they can manage users from the portal's User Management link without being
full Keycloak admins. Keycloak's fine-grained admin permissions cannot be
captured in a realm import, so two manual clicks remain (the
`admin-fine-grained-authz` feature is already enabled on the keycloak service):

1. Log in to http://auth.localhost as `admin`/`admin`, switch to the
   `agentplane` realm.
2. Realm roles -> `user` -> Permissions tab -> enable "Permissions enabled" ->
   open the `map-role` permission -> add a policy that allows the
   `platform-admins` group.
3. Repeat for the `builder` role.

After that, `platform-admins` members can grant exactly `user` and `builder`
(and nothing else) to other accounts.

## Tracing backends

Services (registry, runtime, agentgateway) export OTLP to
`http://otel-collector:4318` — plain, unauthenticated. Credentials for the
backend live only in the collector config under `deploy/compose/otel/`. The
`otel-collector` name is a network alias shared by two collector services, one
per profile — run exactly one of the two profiles at a time:

- **Langfuse (default choice):** `--profile langfuse` starts Langfuse v3
  (web + worker + ClickHouse + Valkey + MinIO) and a collector that forwards
  traces to Langfuse's OTLP endpoint with the bootstrap project keys
  (`pk-lf-local`/`sk-lf-local`).
- **MLflow:** `--profile mlflow` starts an MLflow 3 tracking server (postgres
  backend store) and a collector that forwards traces to MLflow's OTLP
  ingestion (`/v1/traces`, experiment `0`):

  ```bash
  docker compose -f deploy/compose/compose.yaml --profile mlflow up -d --wait
  ```

Switching: `docker compose --profile langfuse down`, then bring the stack up
with the other profile. Without any profile the platform still runs, but no
collector exists and trace exports fail quietly (logged by the SDKs).

## Guard-rails (CORS, rate limits, quotas)

Two protection layers ship on the base stack:

- **Per-owner deployment quota (runtime).**
  `AGENTPLANE_RUNTIME_MAX_DEPLOYMENTS_PER_OWNER` caps how many non-ephemeral
  definitions one owner may keep deployed at once. The base file sets it to
  `25`; `0` means unlimited. Redeploying or rolling back an already-deployed
  definition does not consume a new slot, undeploying frees one, and `admin`
  callers bypass the cap. A deploy over the limit returns `429` with a
  `{"detail": {"error": "deployment_quota_exceeded", ...}}` body (SPEC §7.2).
  Ephemeral playground deploys never count against it.

- **Gateway CORS + rate limiting (agentgateway).**
  `agentgateway/config.yaml` puts a `cors` policy on every browser-facing route
  (`/a2a/*`, `/mcp/*`, `/registry/*`, `/runtime/*`) scoped to the SPA origins
  `http://app.localhost` and `http://builder.localhost`; the SDK/CLI and the
  registry health job are not browsers and are unaffected. `/a2a` and `/mcp`
  additionally carry a `localRateLimit` token bucket (~60 requests/minute,
  state per gateway instance) as a coarse abuse guard. Both policies are
  validated against the agentgateway 0.8.2 route-policy schema. For prod, widen
  the `cors.allowOrigins` to your public app/builder domains when you copy the
  config (see below).

## Production overlay

```bash
docker compose -f deploy/compose/compose.yaml -f deploy/compose/compose.prod.yaml \
  --profile langfuse up -d --wait
```

Set in the environment (or `deploy/compose/.env`): `APP_DOMAIN`, `API_DOMAIN`,
`AUTH_DOMAIN`, `TRACES_DOMAIN`, `BUILDER_DOMAIN`, `ACME_EMAIL`. The overlay
adds a `:443` entrypoint with an ACME (Let's Encrypt) resolver, redirects HTTP
to HTTPS, rewrites every traefik router rule to the public domains, points all
OIDC issuers at `https://$AUTH_DOMAIN/realms/agentplane`, and sets
`KC_HOSTNAME` from `AUTH_DOMAIN`.

**Before exposing this to the internet — all of these are mandatory:**

- **Rotate every bootstrap credential.** The base file ships well-known dev
  values: Keycloak `admin`/`admin`, the demo users, postgres
  `agentplane`/`agentplane`, the `agentplane-cli` client secret,
  `AGENTPLANE_RUNTIME_SECRET_KEY`, the Langfuse `NEXTAUTH_SECRET`/`SALT`/
  `ENCRYPTION_KEY`/init keys, the MinIO, ClickHouse and Valkey passwords.
- **`/v1` stays private.** The overlay's gateway router already excludes it
  (`!PathPrefix(`/v1`)`) — do not loosen that rule; the OpenAI-compatible LLM
  egress must only be reachable by the runtime inside the network.
- **agentgateway config:** `agentgateway/config.yaml` pins the expected `iss`
  claim to `http://auth.localhost/realms/agentplane` and scopes CORS to the
  `*.localhost` SPA origins. For prod, copy it, change the two `issuer` values
  to `https://$AUTH_DOMAIN/realms/agentplane`, replace the `cors.allowOrigins`
  entries with your public app/builder domains (`https://$APP_DOMAIN`,
  `https://$BUILDER_DOMAIN`), and mount the copy over
  `/etc/agentgateway/config.yaml` (the JWKS URL keeps pointing at the internal
  `keycloak:8080` — it never leaves the network).
- **UI config:** `ui/config.json` contains the `*.localhost` URLs; provide a
  prod copy with the public domains and mount it over `/srv/config.json`.
- **`REGISTRY_ALLOW_PRIVATE_URLS` stays `false`** (the overlay sets it
  explicitly) so internal addresses can never leak into the registry.
- **Realm hardening:** switch the realm's `sslRequired` from `none` to
  `external` (Realm settings -> Require SSL), configure real SMTP and enable
  `verifyEmail`, and review `registrationAllowed` for your audience.
- **Keycloak run mode:** the base file uses `start-dev` for the local stack;
  for a real deployment switch the command to `start` (production mode) and
  build an optimized image.
- The compose overlay cannot rewrite the imported realm JSON — realm changes
  after first import are made in the Keycloak admin console (the import only
  seeds a fresh database).

## Smoke test

`./scripts/smoke.sh` is the end-to-end release gate: compose up (langfuse
profile) -> token via client credentials -> deploy the example agent + MCP
flow through the CLI -> call them through the gateway -> assert traces in
Langfuse -> undeploy. See SPEC §7.4.

## Version pins

| Image | Pin |
|---|---|
| `ghcr.io/agentgateway/agentgateway` | `0.8.2` (config verified against its `schema/local.json`) |
| `langfuse/langfuse`, `langfuse/langfuse-worker` | `3.217.0` (keep web + worker in lockstep) |
| `otel/opentelemetry-collector-contrib` | `0.156.0` |
| `ghcr.io/mlflow/mlflow` | `v3.14.0` (OTLP trace ingestion needs >= 3.6) |
| `clickhouse/clickhouse-server` | `25.8` (LTS) |
| `valkey/valkey` | `8.1.8` |
| `minio/minio` | `RELEASE.2025-09-07T16-13-09Z` |

Upgrades land deliberately: agentgateway ships breaking config changes in
minor releases (re-validate `agentgateway/config.yaml` against the new tag's
schema), and Langfuse web/worker must move together.
