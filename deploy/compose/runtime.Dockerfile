FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
RUN uv sync --frozen --no-dev --extra postgres --package agentplane-runtime \
    || uv pip install --system "packages/runtime[postgres]" "packages/core" "packages/sdk"

EXPOSE 8000
CMD ["uv", "run", "--no-dev", "uvicorn", "--factory", "agentplane_runtime.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
