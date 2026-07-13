# agentplane-core

Shared contract of the agentplane platform: the Flow **Definition** schema
(Pydantic v2), registry/validation models, and the `SecretsProvider` /
`VectorStore` / `SearchBackend` interfaces.

- Depends on `pydantic` and `a2a-sdk` only — no I/O, no HTTP, no DB code.
- `agentplane_core.validation.validate_structure()` runs every stateless
  check (stable `E0xx` codes) and is the exact code the runtime uses.
- `python -m agentplane_core.schema_export` emits the public JSON Schema for
  builders and AI-generated definitions.
