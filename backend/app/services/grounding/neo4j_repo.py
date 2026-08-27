"""Neo4j graph repository.

Implements the schema from SYSTEM_REQUIREMENTS section 7::

    (Entity)-[:HAS_FACT]->(Fact)-[:SUPPORTED_BY]->(Document)
    (Entity)-[:CLASSIFIED_AS]->(SensitiveCategory)
    (Policy)-[:FORBIDS]->(SensitiveCategory)

Every driver failure is normalised to :class:`GraphUnavailable`. The grounding service turns
that into ``status=unavailable``, never into "no contradiction found" — silently degrading a
failed lookup into a clean bill of health is exactly what FR-11 forbids.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import Settings
from app.services.grounding.graph_repo import (
    GraphEntity,
    GraphFact,
    GraphHealth,
    GraphRepository,
    GraphUnavailable,
)

logger = logging.getLogger(__name__)

_Q_ENTITIES = """
MATCH (e:Entity)
RETURN e.id AS id, e.name AS name, e.type AS type, e.aliases AS aliases
"""

_Q_FACTS = """
MATCH (e:Entity)-[:HAS_FACT]->(f:Fact)
WHERE e.id IN $entity_ids
OPTIONAL MATCH (f)-[:SUPPORTED_BY]->(d:Document)
RETURN e.id      AS subject,
       f.predicate AS predicate,
       f.object    AS object,
       f.value_type AS value_type,
       f.note      AS note,
       d.id        AS source_document
"""

_Q_DOCUMENTS = """
MATCH (d:Document) RETURN d.id AS id, d.title AS title
"""

_Q_HEALTH = """
MATCH (e:Entity) WITH count(e) AS entities
MATCH (f:Fact)   RETURN entities, count(f) AS facts
"""


class Neo4jGraphRepository(GraphRepository):
    backend = "neo4j"

    def __init__(self, settings: Settings) -> None:
        if not settings.neo4j_configured:
            raise GraphUnavailable(
                "CP_NEO4J_URI / CP_NEO4J_PASSWORD are not set; the in-memory graph will be used."
            )
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise GraphUnavailable(
                "the `neo4j` package is not installed; `pip install -r requirements.txt`"
            ) from exc

        self.settings = settings
        self.database = settings.neo4j_database
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,  # type: ignore[arg-type]
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        #: Entities change rarely in this prototype, so the alias table is fetched once and
        #: reused. This keeps the per-request graph cost to a single facts query.
        self._entity_cache: list[GraphEntity] | None = None

    async def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        try:
            async with self._driver.session(database=self.database) as session:
                result = await session.run(query, **params)
                return [dict(record) async for record in result]
        except Exception as exc:  # noqa: BLE001 - normalise all driver failures
            raise GraphUnavailable(f"Neo4j query failed: {type(exc).__name__}: {exc}") from exc

    async def all_entities(self) -> list[GraphEntity]:
        if self._entity_cache is not None:
            return self._entity_cache
        rows = await self._run(_Q_ENTITIES)
        entities = [
            GraphEntity(
                id=r["id"],
                name=r["name"],
                type=r.get("type") or "Entity",
                aliases=list(r.get("aliases") or []),
            )
            for r in rows
        ]
        self._entity_cache = entities
        return entities

    async def facts_for_entities(self, entity_ids: list[str]) -> list[GraphFact]:
        if not entity_ids:
            return []
        rows = await self._run(_Q_FACTS, entity_ids=list(entity_ids))
        return [
            GraphFact(
                subject=r["subject"],
                predicate=r["predicate"],
                object=str(r["object"]),
                value_type=r.get("value_type") or "string",
                source_document=r.get("source_document"),
                note=r.get("note"),
            )
            for r in rows
        ]

    async def document_titles(self) -> dict[str, str]:
        rows = await self._run(_Q_DOCUMENTS)
        return {r["id"]: r["title"] for r in rows if r.get("id")}

    async def health_check(self) -> GraphHealth:
        try:
            rows = await self._run(_Q_HEALTH)
        except GraphUnavailable as exc:
            return GraphHealth(backend=self.backend, healthy=False, detail=str(exc))
        row = rows[0] if rows else {}
        return GraphHealth(
            backend=self.backend,
            healthy=True,
            detail=f"connected to {self.settings.neo4j_uri}",
            entity_count=row.get("entities"),
            fact_count=row.get("facts"),
        )

    async def aclose(self) -> None:
        await self._driver.close()


__all__ = ["Neo4jGraphRepository"]
