"""FastAPI application factory.

Startup order matters and is deliberate:

1. logging first, so every later message is already PII-filtered;
2. the container next — policies are parsed eagerly, and a malformed profile aborts startup
   rather than letting the server come up with a policy it could not fully read;
3. routers last.

Shutdown closes the graph driver, the Mongo client and the provider client. Failures there are
logged, not raised: a shutdown error should not be the thing that takes the process down.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api.v1 import audits, chat, demo, health, metrics, policies
from app.container import build_container
from app.core.config import Settings, get_settings
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)

DESCRIPTION = """
A Responsible AI control layer. Every model response is evaluated for grounding, cost and
responsibility risk, scored against a named policy profile, and acted on — delivered, masked,
flagged for human review, or withheld — with an audit record explaining why.

All weights, budgets and thresholds are prototype policy configuration, documented as
assumptions in the README. They are not industry standards or calibrated values. Graph evidence
establishes agreement or contradiction with a trusted source; it does not establish truth, and
it says nothing about causality.
""".strip()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = await build_container(settings)
        try:
            yield
        finally:
            container = getattr(app.state, "container", None)
            if container is not None:
                await container.aclose()
            app.state.container = None

    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
    )

    # The dashboard is served from a separate origin in development. Permissive here and
    # deliberately so — this is a local prototype, and a real deployment would pin origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_error_handlers(app)

    for module in (health, chat, audits, metrics, policies, demo):
        app.include_router(module.router, prefix=settings.api_prefix)

    static_dir = Path(__file__).parent / "static"

    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    async def dashboard() -> FileResponse:
        index_file = static_dir / "index.html"
        return FileResponse(index_file)

    return app


app = create_app()

__all__ = ["app", "create_app"]
