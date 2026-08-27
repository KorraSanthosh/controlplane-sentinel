"""Graph backend selection.

Neo4j when ``CP_NEO4J_URI`` and ``CP_NEO4J_PASSWORD`` are set and the database answers,
otherwise the in-memory triple store seeded from ``graph/seed/northwind.json``.

Both backends read the same seed data, which is what makes the parity check in the test suite
meaningful: the six demo scenarios must produce identical grounding results either way. If they
diverge, the Cypher and the in-memory index disagree about the data, and that is a bug worth
failing on.
"""

from __future__ import annotations

import logging

from app.core.config import Settings
from app.services.grounding.graph_repo import GraphRepository, cached_graph_seed
from app.services.grounding.memory_repo import InMemoryGraphRepository

logger = logging.getLogger(__name__)


async def build_graph_repository(settings: Settings) -> GraphRepository:
    seed = cached_graph_seed(str(settings.graph_seed_path_abs))

    if not settings.neo4j_configured:
        logger.info(
            "CP_NEO4J_URI is not set — using the in-memory graph seeded from %s.",
            settings.graph_seed_path_abs.name,
        )
        return InMemoryGraphRepository(seed)

    # Lazy import so the neo4j driver is only required when it is actually configured.
    from app.services.grounding.neo4j_repo import Neo4jGraphRepository

    repo = Neo4jGraphRepository(
        uri=settings.neo4j_uri or "",
        user=settings.neo4j_user,
        password=settings.neo4j_password or "",
        database=settings.neo4j_database,
    )
    health = await repo.health_check()
    if not health.healthy:
        await repo.aclose()
        logger.warning(
            "Neo4j is configured but unreachable (%s) — falling back to the in-memory graph. "
            "Grounding still works; it is just not reading from Aura.",
            health.detail,
        )
        return InMemoryGraphRepository(seed)

    logger.info(
        "Using Neo4j graph backend (%d entit(y/ies) visible).", health.entity_count or 0
    )
    return repo


__all__ = ["build_graph_repository"]
