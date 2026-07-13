FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
RUN uv sync --frozen --no-dev --extra postgres --package agentplane-registry \
    || uv pip install --system "packages/registry[postgres]" "packages/core"

EXPOSE 8100
CMD ["uv", "run", "--no-dev", "uvicorn", "--factory", "agentplane_registry.app:create_app", "--host", "0.0.0.0", "--port", "8100"]
