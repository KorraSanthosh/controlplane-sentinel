"""Audit trail contracts (FR-10).

Storage rules, from NFR-01:

* the raw model response is never persisted — only a SHA-256 hash and a PII-masked,
  truncated preview;
* the same holds for the prompt;
* PII spans are stored as kind + offsets + masked preview, never the matched value.

The hash lets you prove two responses were identical without keeping either.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.decision import Decision, FiredRule
from app.schemas.signals import RiskAssessment, Telemetry


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AuditRecord(BaseModel):
    """One immutable decision record."""

    request_id: str
    timestamp: datetime = Field(default_factory=_utcnow)

    # --- context ---
    use_case: str
    policy_profile: str
    policy_version: str = "0"
    user_id: str | None = None
    tenant_id: str | None = None

    # --- model ---
    provider: str
    model: str

    # --- content references (never raw payloads) ---
    prompt_sha256: str
    prompt_preview: str
    response_sha256: str
    response_preview: str
    delivered_sha256: str
    answer_modified: bool = False

    # --- evaluation ---
    risk: RiskAssessment
    decision: Decision
    reason: str
    fired_rules: list[FiredRule] = Field(default_factory=list)
    unavailable_checks: list[str] = Field(default_factory=list)
    skipped_checks: list[str] = Field(default_factory=list)
    #: Persisted rather than re-derived from ``decision``: a BLOCK on a critical safety rule
    #: also wants human eyes, so the review queue is not simply "everything FLAG".
    requires_human_review: bool = False

    # --- telemetry ---
    telemetry: Telemetry

    def to_document(self) -> dict[str, Any]:
        """Mongo document. ``request_id`` doubles as ``_id`` so writes are idempotent."""
        doc = self.model_dump(mode="json")
        doc["_id"] = self.request_id
        # model_dump(mode="json") stringifies the timestamp. Put a native datetime back so
        # Mongo sorts and range-queries it as a date rather than lexicographically.
        doc["timestamp"] = self.timestamp
        return doc

    @classmethod
    def from_document(cls, doc: dict[str, Any]) -> "AuditRecord":
        payload = {k: v for k, v in doc.items() if k != "_id"}
        return cls.model_validate(payload)


class AuditSummary(BaseModel):
    """Lightweight row for the audit table — avoids shipping full signal payloads."""

    request_id: str
    timestamp: datetime
    use_case: str
    decision: Decision
    overall_risk_score: float
    reason: str
    model: str
    provider: str
    total_ms: float
    controlplane_overhead_ms: float
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    fired_rule_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_record(cls, rec: AuditRecord) -> "AuditSummary":
        return cls(
            request_id=rec.request_id,
            timestamp=rec.timestamp,
            use_case=rec.use_case,
            decision=rec.decision,
            overall_risk_score=rec.risk.overall_score,
            reason=rec.reason,
            model=rec.model,
            provider=rec.provider,
            total_ms=rec.telemetry.total_ms,
            controlplane_overhead_ms=rec.telemetry.controlplane_overhead_ms,
            total_tokens=rec.risk.signals.cost.total_tokens,
            estimated_cost_usd=rec.risk.signals.cost.estimated_cost_usd,
            fired_rule_ids=[r.rule_id for r in rec.fired_rules],
        )


class FeedbackRecord(BaseModel):
    """Human reviewer overriding or confirming a decision (SYSTEM_REQUIREMENTS §6).

    Stored but not yet fed back into scoring — a learning loop is explicitly P2.
    """

    request_id: str
    reviewer: str
    override_decision: Decision | None = None
    agrees_with_decision: bool | None = None
    comment: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class MetricsSummary(BaseModel):
    """Dashboard aggregates (GET /metrics)."""

    total_requests: int = 0
    decision_counts: dict[str, int] = Field(default_factory=dict)
    avg_overall_risk: float = 0.0
    avg_total_ms: float = 0.0
    avg_controlplane_overhead_ms: float = 0.0
    p95_controlplane_overhead_ms: float = 0.0
    deep_path_count: int = 0
    deep_path_rate: float = 0.0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    unavailable_check_counts: dict[str, int] = Field(default_factory=dict)
    top_fired_rules: dict[str, int] = Field(default_factory=dict)


__all__ = [
    "AuditRecord",
    "AuditSummary",
    "FeedbackRecord",
    "MetricsSummary",
]
