#!/usr/bin/env bash
# End-to-end smoke test (SPEC §7.4) — the release gate.
#
#   1. compose up            5. poll registry until healthy
#   2. obtain token          6. A2A message/send "ping" through the gateway
#   3. create LLM resource   7. deploy MCP flow, tools/list + tools/call
#   4. deploy echo agent     8. assert traces in Langfuse
#                            9. undeploy both, entries gone
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

log "1/9 docker compose up (langfuse profile)"
docker compose -f "$COMPOSE_FILE" --profile langfuse up -d --build --wait

log "2/9 obtaining token (client credentials)"
TOKEN=$(curl -fsS "$KEYCLOAK_URL/realms/agentplane/protocol/openid-connect/token" \
  -d grant_type=client_credentials \
  -d client_id=agentplane-cli \
  -d client_secret=agentplane-cli-secret | python -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
[ -n "$TOKEN" ] || fail "no token from keycloak"
export AGENTPLANE_TOKEN="$TOKEN"
export AGENTPLANE_RUNTIME_URL="$RUNTIME_URL"
export AGENTPLANE_REGISTRY_URL="$REGISTRY_URL"

log "3/9 creating LLM resource"
uv run agentplane resources create -f "$EXAMPLES/resources/default-llm.yaml" \
  || echo "  (resource may already exist)"

log "4/9 deploying echo-agent"
DEPLOY_OUT=$(uv run agentplane deploy "$EXAMPLES/echo-agent.yaml")
echo "  $DEPLOY_OUT"
echo "$DEPLOY_OUT" | grep -q "a2a/echo-agent" || fail "deploy did not return an endpoint URL"

log "5/9 polling registry until echo-agent is healthy (<=60s)"
for i in $(seq 1 30); do
  STATUS=$(uv run agentplane search "echo-agent" --json \
    | python -c 'import json,sys; e=[x for x in json.load(sys.stdin) if x["card"]["name"]=="echo-agent"]; print(e[0]["status"] if e else "missing")')
  [ "$STATUS" = "healthy" ] && break
  sleep 2
done
[ "$STATUS" = "healthy" ] || fail "echo-agent never turned healthy (last: $STATUS)"

log "6/9 A2A message/send 'ping' through the gateway"
ANSWER=$(curl -fsS "$GATEWAY_URL/a2a/echo-agent/" \
  -H "Authorization: Bearer $TOKEN" -H "A2A-Version: 1.0" -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"SendMessage","params":{"message":{"messageId":"smoke-1","role":"ROLE_USER","parts":[{"text":"ping"}]}}}' \
  | python -c 'import json,sys; t=json.load(sys.stdin)["result"]["task"]; print(t["artifacts"][0]["parts"][0].get("text",""))')
[ -n "$ANSWER" ] || fail "empty A2A answer"
echo "  answer: $ANSWER"

log "7/9 deploying support-rag (MCP) + tools/list + tools/call"
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

log "8/9 asserting traces in Langfuse"
sleep 5
TRACES=$(curl -fsS -u pk-lf-local:sk-lf-local "$LANGFUSE_URL/api/public/traces?limit=10" \
  | python -c 'import json,sys; print(len(json.load(sys.stdin).get("data", [])))')
[ "$TRACES" -ge 1 ] || fail "no traces in Langfuse"
echo "  $TRACES traces found"

log "9/9 undeploying both flows"
uv run agentplane undeploy echo-agent
uv run agentplane undeploy support-rag
REMAINING=$(uv run agentplane search "" --json | python -c 'import json,sys; print(len(json.load(sys.stdin)))')
[ "$REMAINING" -eq 0 ] || fail "registry entries remain after undeploy"

log "smoke test PASSED"
