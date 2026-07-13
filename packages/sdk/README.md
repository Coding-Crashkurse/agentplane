# agentplane-sdk

Thin typed client + CLI for the agentplane platform. Everything the SDK does
is possible with plain HTTP — the SDK is convenience, not a requirement.

```python
from agentplane_sdk import RuntimeClient

async with RuntimeClient("https://api.example/runtime", token="…") as client:
    info = await client.deploy("support-rag")
    print(info.endpoint_url)
```

CLI:

```
agentplane validate flow.yaml
agentplane deploy flow.yaml [--draft]
agentplane undeploy <name>
agentplane list [--status deployed] [--json]
agentplane export <name> [-o flow.yaml]
agentplane search "invoice" [--tags rag] [--semantic]
agentplane resources list|create -f res.yaml|delete <name>
```

Config resolution: flags → env (`AGENTPLANE_RUNTIME_URL`,
`AGENTPLANE_REGISTRY_URL`, `AGENTPLANE_TOKEN` or OIDC vars) →
`~/.config/agentplane/config.toml`. Exit codes: 0 ok, 1 validation failed,
2 transport/auth error, 3 not found.
