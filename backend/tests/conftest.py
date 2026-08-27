"""Shared fixtures.

Two rules the whole suite depends on:

**No network, ever.** ``_no_network`` patches ``socket.socket`` for the entire session, so a
test that accidentally reaches for Anthropic, Mongo or Neo4j fails loudly instead of quietly
depending on the machine it runs on. The settings fixture also blanks every credential field,
so even the factories' capability checks resolve to the in-process backends.

**Same wiring as production.** Tests build the real :func:`app.container.build_container` rather
than assembling services by hand. If the composition root breaks, the suite notices — which is
the whole reason a composition root exists.
"""

from __future__ import annotations

import socket
from typing import AsyncIterator, Iterator

import pytest
import pytest_asyncio

from app.container import Container, build_container
from app.core.config import Settings
from app.demo.scenarios import ScenarioLibrary, cached_scenario_library
from app.schemas.chat import ChatRequest
from app.services.grounding.graph_repo import cached_graph_seed
from app.services.policy.loader import PolicyRegistry, load_policy_registry


@pytest.fixture(scope="session", autouse=True)
def _no_network() -> Iterator[None]:
    """Make outbound sockets impossible for the duration of the suite."""

    class _Blocked(socket.socket):
        def __init__(self, family=socket.AF_INET, type=socket.SOCK_STREAM, proto=0, fileno=None):
            if family in (getattr(socket, "AF_UNIX", None), getattr(socket, "AF_LOCAL", None)):
                super().__init__(family, type, proto, fileno)
                return
            raise RuntimeError(
                "network access is disabled in tests; the suite must run offline"
            )

    original = socket.socket
    socket.socket = _Blocked  # type: ignore[misc, assignment]
    try:
        yield
    finally:
        socket.socket = original  # type: ignore[misc]


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Credential-free settings: mock provider, in-memory graph, in-memory audit store."""
    return Settings(
        environment="test",
        log_level="WARNING",
        llm_provider="mock",
        anthropic_api_key=None,
        neo4j_uri=None,
        neo4j_password=None,
        mongo_uri=None,
        allow_debug_original=True,
    )


@pytest.fixture(scope="session")
def scenarios(settings: Settings) -> ScenarioLibrary:
    return cached_scenario_library(str(settings.scenarios_path_abs))


@pytest.fixture(scope="session")
def policies(settings: Settings) -> PolicyRegistry:
    return load_policy_registry(settings.policy_dir_abs, settings.default_policy_profile)


@pytest.fixture(scope="session")
def graph_seed(settings: Settings):
    return cached_graph_seed(str(settings.graph_seed_path_abs))


@pytest_asyncio.fixture
async def container(settings: Settings) -> AsyncIterator[Container]:
    c = await build_container(settings)
    try:
        yield c
    finally:
        await c.aclose()


def chat_request(message: str, **kwargs) -> ChatRequest:
    """A ChatRequest with the demo use case pre-set."""
    kwargs.setdefault("use_case", "support_assistant")
    return ChatRequest(message=message, **kwargs)
