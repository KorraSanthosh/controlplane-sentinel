"""Audit storage abstraction.

Two implementations sit behind this interface — MongoDB and an in-memory store — for the same
reason the graph layer has two: the prototype must run, and its tests must pass, with no
infrastructure available (NFR-05). Selection happens once at startup based on whether
``CP_MONGO_URI`` is set.

Filter semantics live in :class:`AuditFilter` rather than being reimplemented per backend, so
"risk score at least 0.5" cannot come to mean two different things depending on where the data
happens to be stored.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.schemas.audit import AuditRecord, FeedbackRecord
from app.schemas.decision import Decision


class AuditUnavailable(RuntimeError):
    """The audit store could not be reached."""


@dataclass(frozen=True)
class AuditHealth:
    backend: str
    available: bool
    detail: str = ""
    record_count: int | None = None


@dataclass(frozen=True)
class AuditFilter:
    """Query filter, defined once for every backend."""

    decision: Decision | None = None
    use_case: str | None = None
    min_risk: float | None = None
    requires_human_review: bool | None = None

    def matches(self, record: AuditRecord) -> bool:
        if self.decision is not None and record.decision is not self.decision:
            return False
        if self.use_case is not None and record.use_case != self.use_case:
            return False
        if self.min_risk is not None and record.risk.overall_score < self.min_risk:
            return False
        if self.requires_human_review is not None:
            if record.requires_human_review is not self.requires_human_review:
                return False
        return True

    def to_query(self) -> dict[str, Any]:
        query: dict[str, Any] = {}
        if self.decision is not None:
            query["decision"] = self.decision.value
        if self.use_case is not None:
            query["use_case"] = self.use_case
        if self.min_risk is not None:
            query["risk.overall_score"] = {"$gte": self.min_risk}
        if self.requires_human_review is not None:
            query["requires_human_review"] = self.requires_human_review
        return query


EMPTY_FILTER = AuditFilter()


class AuditRepository(ABC):
    """Storage for decision records and reviewer feedback."""

    #: "mongo" | "memory" — surfaced on /health so the deployment is never a mystery.
    backend: str = "unknown"

    @abstractmethod
    async def save(self, record: AuditRecord) -> None:
        """Persist one record. Idempotent on ``request_id``."""

    @abstractmethod
    async def get(self, request_id: str) -> AuditRecord | None: ...

    @abstractmethod
    async def list(
        self,
        *,
        filter_: AuditFilter = EMPTY_FILTER,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AuditRecord]:
        """Newest first."""

    @abstractmethod
    async def count(self, *, filter_: AuditFilter = EMPTY_FILTER) -> int: ...

    @abstractmethod
    async def save_feedback(self, feedback: FeedbackRecord) -> None: ...

    @abstractmethod
    async def list_feedback(self, request_id: str) -> list[FeedbackRecord]: ...

    @abstractmethod
    async def health_check(self) -> AuditHealth: ...

    async def aclose(self) -> None:  # pragma: no cover - overridden where needed
        return None


__all__ = [
    "EMPTY_FILTER",
    "AuditFilter",
    "AuditHealth",
    "AuditRepository",
    "AuditUnavailable",
]
