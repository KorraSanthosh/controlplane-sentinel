"""Audit backend selection.

Mongo when ``CP_MONGO_URI`` is set and reachable, in-memory otherwise. The fallback is
deliberate and loud: the prototype must be runnable from a clean checkout with no Atlas
account, and ``/health`` always reports which backend is actually in use so nobody mistakes an
ephemeral store for a durable one.
"""

from __future__ import annotations

import logging

from app.core.config import Settings
from app.services.audit.memory_repo import InMemoryAuditRepository
from app.services.audit.repository import AuditRepository

logger = logging.getLogger(__name__)


async def build_audit_repository(settings: Settings) -> AuditRepository:
    if not settings.mongo_configured:
        logger.info("CP_MONGO_URI is not set — using the in-memory audit store.")
        return InMemoryAuditRepository()

    # Imported lazily so a checkout without pymongo installed still runs on the memory path.
    from app.services.audit.mongo_repo import MongoAuditRepository

    repo = MongoAuditRepository(settings.mongo_uri or "", settings.mongo_db)
    health = await repo.health_check()
    if not health.available:
        await repo.aclose()
        logger.warning(
            "MongoDB is configured but unreachable (%s) — falling back to the in-memory audit "
            "store. Records will not survive a restart.",
            health.detail,
        )
        return InMemoryAuditRepository()

    await repo.ensure_indexes()
    logger.info("Using MongoDB audit store (database '%s').", settings.mongo_db)
    return repo


__all__ = ["build_audit_repository"]
