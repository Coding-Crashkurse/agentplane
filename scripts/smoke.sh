#!/usr/bin/env bash
# End-to-end smoke test (SPEC §7.4) — the release gate.
#
#   1. compose up            6. A2A message/send "ping" through the gateway
#   2. obtain token          7. orchestrator deploy (registry-resolved sub-agent) + E062 check
#   3. create LLM resource   8. deploy MCP flow, tools/list + tools/call
#   4. deploy echo agent     9. assert traces in Langfuse
#   5. poll until healthy   10. undeploy all, entries gone
set -euo pipefail

COMPOSE_FILE="$(dirname "$0")/../deploy/compose/compose.yaml"
GATEWAY_URL="${GATEWAY_URL:-http://api.localhost}"
KEYCLOAK_URL="${KEYCLOAK_URL:-http://auth.localhost}"
LANGFUSE_URL="${LANGFUSE_URL:-http://langfuse.localhost}"
RUNTIME_URL="${AGENTPLANE_RUNTIME_URL:-$GATEWAY_URL/runtime}"
REGISTRY_URL="${AGENTPLANE_REGISTRY_URL:-$GATEWAY_URL/registry}"
EXAMPLES="$(dirname "$0")/../examples"

log() { printf '\n\033[1;34m[smoke]\033[0m %s\n' "$*"; }
fail() { printf '\n\033[1;31m[smoke] FAILED:\033[0m %s\n' "$*"; exit 1; }

log "1/10 docker compose up (langfuse profile)"
docker compose -f "$COMPOSE_FILE" --profile langfuse up -d --build --wait

log "2/10 obtaining token (client credentials)"
TOKEN=$(curl -fsS "$KEYCLOAK_URL/realms/agentplane/protocol/openid-connect/token" \
  -d grant_type=client_credentials \
  -d client_id=agentplane-cli \
  -d client_secret=agentplane-cli-secret | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
[ -n "$TOKEN" ] || fail "no token from keycloak"
export AGENTPLANE_TOKEN="$TOKEN"
export AGENTPLANE_RUNTIME_URL="$RUNTIME_URL"
export AGENTPLANE_REGISTRY_URL="$REGISTRY_URL"

log "3/10 creating LLM resource"
uv run agentplane resources create -f "$EXAMPLES/resources/default-llm.yaml" \
  || echo "  (resource may already exist)"

log "4/10 deploying echo-agent"
DEPLOY_OUT=$(uv run agentplane deploy "$EXAMPLES/echo-agent.yaml")
echo "  $DEPLOY_OUT"
echo "$DEPLOY_OUT" | grep -q "a2a/echo-agent" || fail "deploy did not return an endpoint URL"

log "5/10 polling registry until echo-agent is healthy (<=60s)"
for i in $(seq 1 30); do
  STATUS=$(uv run agentplane search "echo-agent" --json \
    | python -c 'import json,sys; e=[x for x in json.load(sys.stdin) if x["card"]["name"]=="echo-agent"]; print(e[0]["status"] if e else "missing")')
  [ "$STATUS" = "healthy" ] && break
  sleep 2
done
[ "$STATUS" = "healthy" ] || fail "echo-agent never turned healthy (last: $STATUS)"

log "6/10 A2A message/send 'ping' through the gateway"
ANSWER=$(curl -fsS "$GATEWAY_URL/a2a/echo-agent/" \
  -H "Authorization: Bearer $TOKEN" -H "A2A-Version: 1.0" -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":{"messageId":"smoke-1","role":"ROLE_USER","parts":[{"text":"ping"}]}}}' \
  | python -c 'import json,sys; t=json.load(sys.stdin)["result"]["task"]; print(t["artifacts"][0]["parts"][0].get("text",""))')
[ -n "$ANSWER" ] || fail "empty A2A answer"
echo "  answer: $ANSWER"

log "7/10 orchestrator: registry-resolved sub-agent deploys; ghost reference fails E062"
TMP_DIR=$(mktemp -d)
cat > "$TMP_DIR/orchestrator-smoke.yaml" <<'YAML'
schema_version: 1
name: orchestrator-smoke
description: "Smoke: orchestrator delegating to echo-agent"
expose:
  kind: a2a
nodes:
  - id: start_1
    type: start
    version: 1
    config:
      input_schema:
        type: object
        properties:
          request: { type: string }
        required: [request]
  - id: agent_1
    type: agent
    version: 1
    config:
      resource: default-llm
      prompt: "{request}"
      agents:
        - name: echo-agent
      max_iterations: 3
  - id: end_1
    type: end
    version: 1
    config:
      output_from: agent_1.text
edges:
  - { from: start_1.request, to: agent_1.request }
  - { from: agent_1.text, to: end_1.input }
YAML
uv run agentplane deploy "$TMP_DIR/orchestrator-smoke.yaml" \
  | grep -q "a2a/orchestrator-smoke" || fail "orchestrator deploy did not return an endpoint URL"
sed 's/name: echo-agent/name: ghost-agent/; s/name: orchestrator-smoke/name: orchestrator-ghost/' \
  "$TMP_DIR/orchestrator-smoke.yaml" > "$TMP_DIR/orchestrator-ghost.yaml"
if uv run agentplane validate "$TMP_DIR/orchestrator-ghost.yaml" --remote --json \
  > "$TMP_DIR/ghost.json"; then
  fail "orchestrator with unknown sub-agent unexpectedly validated"
fi
grep -q "E062" "$TMP_DIR/ghost.json" || fail "expected E062 for the unknown agent reference"
echo "  orchestrator deployed; ghost reference correctly rejected (E062)"

log "8/10 deploying support-rag (MCP) + tools/list + tools/call"
uv run agentplane resources create -f "$EXAMPLES/resources/kb-support.yaml" \
  || echo "  (resource may already exist)"
uv run agentplane deploy "$EXAMPLES/support-rag.yaml"
uv run python - <<PY
import asyncio, os
from fastmcp import Client

async def main() -> None:
    client = Client("$GATEWAY_URL/mcp/support-rag/", auth=os.environ["AGENTPLANE_TOKEN"])
    async with client:
        tools = await client.list_tools()
        assert any(t.name == "search_support_kb" for t in tools), tools
        result = await client.call_tool("search_support_kb", {"query": "how do I reset?"})
        text = result.content[0].text if result.content else ""
        assert text, "empty MCP tool result"
        print("  tool answer:", text[:120])

asyncio.run(main())
PY

log "9/10 asserting traces in Langfuse"
sleep 5
TRACES=$(curl -fsS -u pk-lf-local:sk-lf-local "$LANGFUSE_URL/api/public/traces?limit=10" \
  | python -c 'import json,sys; print(len(json.load(sys.stdin).get("data", [])))')
[ "$TRACES" -ge 1 ] || fail "no traces in Langfuse"
echo "  $TRACES traces found"

log "10/10 undeploying all flows"
uv run agentplane undeploy orchestrator-smoke
uv run agentplane undeploy echo-agent
uv run agentplane undeploy support-rag
REMAINING=$(uv run agentplane search "" --json | python -c 'import json,sys; print(len(json.load(sys.stdin)))')
[ "$REMAINING" -eq 0 ] || fail "registry entries remain after undeploy"

log "smoke test PASSED"
