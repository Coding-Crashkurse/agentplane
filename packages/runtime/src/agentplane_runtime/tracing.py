"""OpenTelemetry wiring — OTLP endpoint is configuration, never a vendor SDK.

Uses the standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` variable; without it the
no-op tracer provider stays active and nothing is exported.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from opentelemetry import propagate, trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode


def setup_tracing(app: FastAPI, *, service_name: str) -> None:
    """Install an OTLP exporter (when configured) and a request span middleware."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)

    tracer = trace.get_tracer(service_name)

    @app.middleware("http")
    async def _trace_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Continue the caller's trace (SPEC §12): the gateway forwards the
        # browser's traceparent; without extraction every service would start
        # its own disconnected trace and the chat's trace link would miss the
        # flow and LLM spans.
        with tracer.start_as_current_span(
            f"{request.method} {request.url.path}",
            context=propagate.extract(dict(request.headers)),
            attributes={
                "http.request.method": request.method,
                "url.path": request.url.path,
            },
        ) as span:
            response = await call_next(request)
            span.set_attribute("http.response.status_code", response.status_code)
            if response.status_code >= 500:  # noqa: PLR2004
                span.set_status(Status(StatusCode.ERROR))
            return response


__all__ = ["setup_tracing"]
