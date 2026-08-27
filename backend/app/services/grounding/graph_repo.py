"""Graph repository abstraction for grounding evidence.

Two implementations sit behind this interface:

* :class:`app.services.grounding.neo4j_repo.Neo4jGraphRepository` — real Cypher, pointed at
  Neo4j Aura or any bolt endpoint;
* :class:`app.services.grounding.memory_repo.InMemoryGraphRepository` — deterministic,
  in-process, seeded from the same JSON file.

Both load ``graph/seed/northwind.json``, so grounding results are identical either way. That
is what lets the test suite and a laptop with no database still exercise the full pipeline
(NFR-05), while ``/health`` reports which backend is actually live so the fallback is never
silent.
"""

from __future__ import annotations

import abc
import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class GraphEntity(BaseModel):
    id: str
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)

    def alias_set(self) -> set[str]:
        """Lowercased aliases plus the display name."""
        return {a.lower() for a in [*self.aliases, self.name]}


class GraphFact(BaseModel):
    subject: str
    predicate: str
    object: str
    value_type: str = "string"
    source_document: str | None = None
    note: str | None = None

    def reference(self) -> str:
        """Compact human-readable citation, e.g. ``plan_premium-INCLUDES_ROAMING-false``."""
        return f"{self.subject}-{self.predicate}-{self.object}"


class GraphDocument(BaseModel):
    id: str
    title: str
    reference: str | None = None


class SensitiveCategory(BaseModel):
    id: str
    name: str
    description: str = ""


class PolicyNode(BaseModel):
    id: str
    name: str
    forbids: list[str] = Field(default_factory=list)
    source_document: str | None = None


class GraphSeed(BaseModel):
    """Parsed contents of the seed file."""

    version: int = 1
    domain: str = "unknown"
    description: str = ""
    documents: list[GraphDocument] = Field(default_factory=list)
    entities: list[GraphEntity] = Field(default_factory=list)
    facts: list[GraphFact] = Field(default_factory=list)
    sensitive_categories: list[SensitiveCategory] = Field(default_factory=list)
    policy_nodes: list[PolicyNode] = Field(default_factory=list)


class GraphHealth(BaseModel):
    backend: str
    healthy: bool
    detail: str = ""
    entity_count: int | None = None
    fact_count: int | None = None


class GraphUnavailable(Exception):
    """The graph could not be queried.

    Raised rather than returning an empty result set, because "no evidence found" and
    "could not look for evidence" must lead to different decisions (FR-11).
    """


def load_graph_seed(path: str | Path) -> GraphSeed:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"graph seed not found at {p}")
    return GraphSeed.model_validate(json.loads(p.read_text(encoding="utf-8")))


@lru_cache(maxsize=4)
def cached_graph_seed(path: str) -> GraphSeed:
    return load_graph_seed(path)


class GraphRepository(abc.ABC):
    """Read interface used by the grounding service."""

    backend: str = "base"

    @abc.abstractmethod
    async def all_entities(self) -> list[GraphEntity]:
        """Every entity with its aliases — used to spot entity mentions in a response."""

    @abc.abstractmethod
    async def facts_for_entities(self, entity_ids: list[str]) -> list[GraphFact]:
        """All facts for the given subjects, in one round trip."""

    @abc.abstractmethod
    async def health_check(self) -> GraphHealth:
        """Report reachability and rough size."""

    @abc.abstractmethod
    async def document_titles(self) -> dict[str, str]:
        """Map document id → title, for citation display."""

    async def aclose(self) -> None:
        return None


__all__ = [
    "GraphDocument",
    "GraphEntity",
    "GraphFact",
    "GraphHealth",
    "GraphRepository",
    "GraphSeed",
    "GraphUnavailable",
    "PolicyNode",
    "SensitiveCategory",
    "cached_graph_seed",
    "load_graph_seed",
]
