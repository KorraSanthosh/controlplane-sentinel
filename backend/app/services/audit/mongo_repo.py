"""MongoDB audit store.

Uses PyMongo's native async client (``AsyncMongoClient``, PyMongo 4.13+). Motor is deprecated,
so this is the supported async path rather than a legacy one.

Every driver failure is normalised to :class:`AuditUnavailable`. Callers must never have to
know about ``pymongo.errors`` — that is the whole point of the repository boundary, and it also
means the audit service can apply one uniform policy: a storage outage degrades the audit trail
loudly but never fails the user's request.
"""

from __future__ import annotations

import logging
from typing import Any

from pymongo import AsyncMongoClient, DESCENDING
from pymongo.errors import PyMongoError

from app.schemas.audit import AuditRecord, FeedbackRecord
from app.services.audit.repository import (
    EMPTY_FILTER,
    AuditFilter,
    AuditHealth,
    AuditRepository,
    AuditUnavailable,
)

logger = logging.getLogger(__name__)

AUDIT_COLLECTION = "audit_records"
FEEDBACK_COLLECTION = "feedback"


class MongoAuditRepository(AuditRepository):
    backend = "mongo"

    def __init__(
        self,
        uri: str,
        database: str = "controlplane",
        *,
        timeout_ms: int = 5000,
    ) -> None:
        self._client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(
            uri,
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
            tz_aware=True,
        )
        self._db = self._client[database]
        self.database = database
        self._indexes_ready = False

    @property
    def _audits(self) -> Any:
        return self._db[AUDIT_COLLECTION]

    @property
    def _feedback(self) -> Any:
        return self._db[FEEDBACK_COLLECTION]

    async def ensure_indexes(self) -> None:
        """Create the indexes the dashboard queries rely on.

        Called from application startup. Failure is logged, not raised: missing indexes make
        queries slower, and refusing to serve traffic over that would be the wrong trade.
        """
        if self._indexes_ready:
            return
        try:
            await self._audits.create_index([("timestamp", DESCENDING)])
            await self._audits.create_index([("decision", 1), ("timestamp", DESCENDING)])
            await self._audits.create_index([("use_case", 1), ("timestamp", DESCENDING)])
            await self._audits.create_index([("requires_human_review", 1)])
            await self._feedback.create_index([("request_id", 1)])
            self._indexes_ready = True
            logger.info("Mongo audit indexes ready on %s.%s", self.database, AUDIT_COLLECTION)
        except PyMongoError as exc:
            logger.warning("Could not create Mongo audit indexes: %s", exc)

    async def save(self, record: AuditRecord) -> None:
        doc = record.to_document()
        try:
            # Upsert on _id (== request_id) so a retried background write cannot duplicate a
            # decision record.
            await self._audits.replace_one({"_id": record.request_id}, doc, upsert=True)
        except PyMongoError as exc:
            raise AuditUnavailable(f"Mongo write failed: {exc}") from exc

    async def get(self, request_id: str) -> AuditRecord | None:
        try:
            doc = await self._audits.find_one({"_id": request_id})
        except PyMongoError as exc:
            raise AuditUnavailable(f"Mongo read failed: {exc}") from exc
        return AuditRecord.from_document(doc) if doc else None

    async def list(
        self,
        *,
        filter_: AuditFilter = EMPTY_FILTER,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AuditRecord]:
        try:
            cursor = (
                self._audits.find(filter_.to_query())
                .sort("timestamp", DESCENDING)
                .skip(max(0, offset))
                .limit(max(0, limit))
            )
            docs = [doc async for doc in cursor]
        except PyMongoError as exc:
            raise AuditUnavailable(f"Mongo query failed: {exc}") from exc
        return [AuditRecord.from_document(doc) for doc in docs]

    async def count(self, *, filter_: AuditFilter = EMPTY_FILTER) -> int:
        try:
            return await self._audits.count_documents(filter_.to_query())
        except PyMongoError as exc:
            raise AuditUnavailable(f"Mongo count failed: {exc}") from exc

    async def save_feedback(self, feedback: FeedbackRecord) -> None:
        try:
            await self._feedback.insert_one(feedback.model_dump(mode="json"))
        except PyMongoError as exc:
            raise AuditUnavailable(f"Mongo feedback write failed: {exc}") from exc

    async def list_feedback(self, request_id: str) -> list[FeedbackRecord]:
        try:
            cursor = self._feedback.find({"request_id": request_id}).sort("created_at", 1)
            docs = [doc async for doc in cursor]
        except PyMongoError as exc:
            raise AuditUnavailable(f"Mongo feedback query failed: {exc}") from exc
        return [
            FeedbackRecord.model_validate({k: v for k, v in doc.items() if k != "_id"})
            for doc in docs
        ]

    async def health_check(self) -> AuditHealth:
        try:
            await self._db.command("ping")
            count = await self._audits.count_documents({})
        except PyMongoError as exc:
            return AuditHealth(backend=self.backend, available=False, detail=str(exc))
        return AuditHealth(
            backend=self.backend,
            available=True,
            detail=f"Connected to database '{self.database}'.",
            record_count=count,
        )

    async def aclose(self) -> None:
        await self._client.close()


__all__ = ["AUDIT_COLLECTION", "FEEDBACK_COLLECTION", "MongoAuditRepository"]
