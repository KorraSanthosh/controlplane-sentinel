"""In-memory audit store.

The fallback when ``CP_MONGO_URI`` is unset, and what the whole test suite runs against. It is
a real implementation of the interface, not a stub: the same filtering, ordering and pagination
semantics as the Mongo backend, so a test that passes here means something.

Two honest limitations, both fine for a prototype and both stated on ``/health``: records are
lost on restart, and the store is capped so a long demo cannot exhaust memory.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict

from app.schemas.audit import AuditRecord, FeedbackRecord
from app.services.audit.repository import (
    EMPTY_FILTER,
    AuditFilter,
    AuditHealth,
    AuditRepository,
)

logger = logging.getLogger(__name__)

#: Ring-buffer bound. Oldest records are evicted first.
DEFAULT_MAX_RECORDS = 5000


class InMemoryAuditRepository(AuditRepository):
    backend = "memory"

    def __init__(self, max_records: int = DEFAULT_MAX_RECORDS) -> None:
        self.max_records = max_records
        # Insertion-ordered so "newest first" is just a reversal — no sort key needed, and
        # ties on identical timestamps stay stable.
        self._records: OrderedDict[str, AuditRecord] = OrderedDict()
        self._feedback: dict[str, list[FeedbackRecord]] = {}
        self._lock = asyncio.Lock()
        self._evicted = 0

    async def save(self, record: AuditRecord) -> None:
        async with self._lock:
            # Re-inserting moves the key to the end, matching Mongo's upsert semantics where a
            # replaced record is still the most recent version of that request.
            self._records.pop(record.request_id, None)
            self._records[record.request_id] = record
            while len(self._records) > self.max_records:
                self._records.popitem(last=False)
                self._evicted += 1

    async def get(self, request_id: str) -> AuditRecord | None:
        return self._records.get(request_id)

    async def list(
        self,
        *,
        filter_: AuditFilter = EMPTY_FILTER,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AuditRecord]:
        matching = [r for r in reversed(self._records.values()) if filter_.matches(r)]
        return matching[offset : offset + limit]

    async def count(self, *, filter_: AuditFilter = EMPTY_FILTER) -> int:
        if filter_ == EMPTY_FILTER:
            return len(self._records)
        return sum(1 for r in self._records.values() if filter_.matches(r))

    async def save_feedback(self, feedback: FeedbackRecord) -> None:
        async with self._lock:
            self._feedback.setdefault(feedback.request_id, []).append(feedback)

    async def list_feedback(self, request_id: str) -> list[FeedbackRecord]:
        return list(self._feedback.get(request_id, ()))

    async def health_check(self) -> AuditHealth:
        detail = (
            "In-memory audit store: records do not survive a restart. Set CP_MONGO_URI for "
            "durable persistence."
        )
        if self._evicted:
            detail += f" {self._evicted} record(s) evicted at the {self.max_records} cap."
        return AuditHealth(
            backend=self.backend,
            available=True,
            detail=detail,
            record_count=len(self._records),
        )


__all__ = ["DEFAULT_MAX_RECORDS", "InMemoryAuditRepository"]
