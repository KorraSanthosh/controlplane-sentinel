"""In-memory graph repository.

Seeded from the same JSON as Neo4j, so grounding behaviour is identical. This is the
fallback that keeps the app and the whole test suite runnable with no database.
"""

from __future__ import annotations

from app.services.grounding.graph_repo import (
    GraphEntity,
    GraphFact,
    GraphHealth,
    GraphRepository,
    GraphSeed,
)


class InMemoryGraphRepository(GraphRepository):
    backend = "memory"

    def __init__(self, seed: GraphSeed) -> None:
        self.seed = seed
        self._facts_by_subject: dict[str, list[GraphFact]] = {}
        for fact in seed.facts:
            self._facts_by_subject.setdefault(fact.subject, []).append(fact)
        self._doc_titles = {d.id: d.title for d in seed.documents}

    async def all_entities(self) -> list[GraphEntity]:
        return list(self.seed.entities)

    async def facts_for_entities(self, entity_ids: list[str]) -> list[GraphFact]:
        out: list[GraphFact] = []
        for eid in entity_ids:
            out.extend(self._facts_by_subject.get(eid, ()))
        return out

    async def document_titles(self) -> dict[str, str]:
        return dict(self._doc_titles)

    async def health_check(self) -> GraphHealth:
        return GraphHealth(
            backend=self.backend,
            healthy=True,
            detail=f"in-process seed ({self.seed.domain}); no external dependency",
            entity_count=len(self.seed.entities),
            fact_count=len(self.seed.facts),
        )


__all__ = ["InMemoryGraphRepository"]
