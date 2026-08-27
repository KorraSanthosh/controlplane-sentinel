"""Audit service (FR-10) — building, storing and aggregating decision records.

Two responsibilities worth calling out.

**Content minimisation.** No raw prompt or response ever reaches storage. Each is reduced to a
SHA-256 hash plus a PII-masked, truncated preview. The hash still lets you prove two responses
were byte-identical, or match a record against a response you already hold, without the store
itself becoming a place personal data accumulates (NFR-01).

**Storage failure is not request failure.** Persisting is called off the response path via
FastAPI ``BackgroundTasks``. If the store is down, the user still gets their answer and the gap
is logged at ERROR — but the decision was still *made* and *returned*, so the control layer did
its job. The alternative (500 on an audit outage) would make the governance layer the single
point of failure for the product it protects. The trade is recorded here rather than left
implicit; a deployment that requires guaranteed audit durability should write synchronously and
fail closed instead.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter

from app.schemas.audit import AuditRecord, AuditSummary, FeedbackRecord, MetricsSummary
from app.schemas.chat import ChatRequest
from app.schemas.decision import DecisionResult
from app.schemas.signals import RiskAssessment, Telemetry
from app.services.audit.repository import (
    EMPTY_FILTER,
    AuditFilter,
    AuditHealth,
    AuditRepository,
    AuditUnavailable,
)
from app.services.pii.service import PIIService

logger = logging.getLogger(__name__)

#: How much of a prompt/response preview is kept. Enough to recognise the exchange in the
#: dashboard, not enough to be a copy of it.
PREVIEW_CHARS = 280

#: Metrics are computed over the most recent N records rather than the whole collection. A
#: prototype-scale simplification: correct for the demo, and the point where a real deployment
#: would move to a server-side aggregation pipeline or a rollup collection.
METRICS_WINDOW = 1000


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class AuditService:
    name = "audit"

    def __init__(
        self,
        repo: AuditRepository,
        pii: PIIService,
        *,
        preview_chars: int = PREVIEW_CHARS,
        metrics_window: int = METRICS_WINDOW,
    ) -> None:
        self.repo = repo
        self.pii = pii
        self.preview_chars = preview_chars
        self.metrics_window = metrics_window

    # -- construction ------------------------------------------------------
    def preview(self, text: str) -> str:
        """Mask any PII, then truncate. Both, in that order — truncating first could cut a
        span in half and leave a partial identifier behind."""
        return self.pii.mask_for_storage(text or "", limit=self.preview_chars)

    def build_record(
        self,
        *,
        request_id: str,
        request: ChatRequest,
        prompt: str,
        original_response: str,
        delivered_response: str,
        provider: str,
        model: str,
        assessment: RiskAssessment,
        decision: DecisionResult,
        telemetry: Telemetry,
    ) -> AuditRecord:
        return AuditRecord(
            request_id=request_id,
            use_case=request.use_case,
            policy_profile=decision.policy_profile,
            policy_version=decision.policy_version,
            user_id=request.user_id,
            tenant_id=request.tenant_id,
            provider=provider,
            model=model,
            prompt_sha256=sha256_text(prompt),
            prompt_preview=self.preview(prompt),
            response_sha256=sha256_text(original_response),
            response_preview=self.preview(original_response),
            delivered_sha256=sha256_text(delivered_response),
            answer_modified=delivered_response != original_response,
            risk=assessment,
            decision=decision.decision,
            reason=decision.reason,
            fired_rules=decision.fired_rules,
            unavailable_checks=assessment.unavailable_checks,
            skipped_checks=assessment.skipped_checks,
            requires_human_review=decision.requires_human_review,
            telemetry=telemetry,
        )

    # -- persistence -------------------------------------------------------
    async def persist(self, record: AuditRecord) -> bool:
        """Store a record. Returns success; never raises.

        Runs as a background task, where an exception would be swallowed by the event loop
        anyway. Better to catch it here and say so in the log.
        """
        try:
            await self.repo.save(record)
            return True
        except AuditUnavailable as exc:
            logger.error(
                "AUDIT GAP: decision %s for request %s was returned to the caller but not "
                "persisted (%s)",
                record.decision.value,
                record.request_id,
                exc,
            )
        except Exception:  # noqa: BLE001 - a background task must not die silently
            logger.exception(
                "AUDIT GAP: unexpected failure persisting request %s", record.request_id
            )
        return False

    async def save_feedback(self, feedback: FeedbackRecord) -> None:
        await self.repo.save_feedback(feedback)

    # -- reads -------------------------------------------------------------
    async def get(self, request_id: str) -> AuditRecord | None:
        return await self.repo.get(request_id)

    async def list_summaries(
        self,
        *,
        filter_: AuditFilter = EMPTY_FILTER,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[AuditSummary], int]:
        records = await self.repo.list(filter_=filter_, limit=limit, offset=offset)
        total = await self.repo.count(filter_=filter_)
        return [AuditSummary.from_record(r) for r in records], total

    async def list_feedback(self, request_id: str) -> list[FeedbackRecord]:
        return await self.repo.list_feedback(request_id)

    async def health_check(self) -> AuditHealth:
        return await self.repo.health_check()

    # -- aggregation -------------------------------------------------------
    async def metrics(self, *, filter_: AuditFilter = EMPTY_FILTER) -> MetricsSummary:
        records = await self.repo.list(filter_=filter_, limit=self.metrics_window, offset=0)
        if not records:
            return MetricsSummary()

        overheads = sorted(r.telemetry.controlplane_overhead_ms for r in records)
        n = len(records)

        decision_counts = Counter(r.decision.value for r in records)
        unavailable_counts: Counter[str] = Counter()
        fired_counts: Counter[str] = Counter()
        for record in records:
            unavailable_counts.update(record.unavailable_checks)
            fired_counts.update(r.rule_id for r in record.fired_rules)

        deep = sum(1 for r in records if r.telemetry.path_taken == "deep")
        tokens = sum(r.risk.signals.cost.total_tokens or 0 for r in records)
        spend = sum(r.risk.signals.cost.estimated_cost_usd or 0.0 for r in records)

        return MetricsSummary(
            total_requests=n,
            decision_counts=dict(decision_counts),
            avg_overall_risk=round(sum(r.risk.overall_score for r in records) / n, 4),
            avg_total_ms=round(sum(r.telemetry.total_ms for r in records) / n, 2),
            avg_controlplane_overhead_ms=round(sum(overheads) / n, 2),
            p95_controlplane_overhead_ms=round(_percentile(overheads, 0.95), 2),
            deep_path_count=deep,
            deep_path_rate=round(deep / n, 4),
            total_tokens=tokens,
            estimated_cost_usd=round(spend, 6),
            unavailable_check_counts=dict(unavailable_counts),
            top_fired_rules=dict(fired_counts.most_common(10)),
        )


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile. Exact for the small windows this operates on."""
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[index]


__all__ = ["METRICS_WINDOW", "PREVIEW_CHARS", "AuditService", "sha256_text"]
