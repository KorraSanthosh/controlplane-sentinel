"""Signal and assessment builders shared by the unit tests.

Kept out of ``conftest.py`` because these are plain constructors, not fixtures — the policy and
decision tests need to call them repeatedly with different arguments inside one test.
"""

from __future__ import annotations

from app.schemas.signals import (
    BiasSignal,
    CostSignal,
    GroundingSignal,
    GroundingStatus,
    PIIKind,
    PIIMatch,
    PIISignal,
    PolicyViolation,
    RiskAssessment,
    RiskSignals,
    SafetySignal,
    Severity,
    SignalStatus,
)
from app.services.risk.scoring import RiskScorer

_scorer = RiskScorer()


def grounding_signal(
    *,
    status: SignalStatus = SignalStatus.PASS,
    grounding_status: GroundingStatus = GroundingStatus.GROUNDED,
    score: float = 0.0,
    severity: Severity = Severity.NONE,
    error: str | None = None,
    **kwargs,
) -> GroundingSignal:
    return GroundingSignal(
        status=status,
        grounding_status=grounding_status,
        score=score,
        severity=severity,
        error=error,
        **kwargs,
    )


def pii_signal(
    *,
    status: SignalStatus = SignalStatus.PASS,
    score: float = 0.0,
    severity: Severity = Severity.NONE,
    detected: bool = False,
    redactable: bool = False,
    kinds: tuple[PIIKind, ...] = (),
    error: str | None = None,
) -> PIISignal:
    matches = [
        PIIMatch(kind=kind, start=i * 10, end=i * 10 + 5, preview="*****")
        for i, kind in enumerate(kinds)
    ]
    counts: dict[str, int] = {}
    for kind in kinds:
        counts[kind.value] = counts.get(kind.value, 0) + 1
    return PIISignal(
        status=status,
        score=score,
        severity=severity,
        detected=detected,
        redactable=redactable,
        matches=matches,
        counts=counts,
        error=error,
    )


def safety_signal(
    *,
    status: SignalStatus = SignalStatus.PASS,
    score: float = 0.0,
    severity: Severity = Severity.NONE,
    categories: tuple[str, ...] = (),
    judge_used: bool = False,
    error: str | None = None,
) -> SafetySignal:
    violations = [
        PolicyViolation(
            policy_id=f"rule_{cat}", category=cat, severity=severity, reason="test violation"
        )
        for cat in categories
    ]
    return SafetySignal(
        status=status,
        score=score,
        severity=severity,
        violations=violations,
        judge_used=judge_used,
        error=error,
    )


def bias_signal(
    *,
    status: SignalStatus = SignalStatus.PASS,
    score: float = 0.0,
    severity: Severity = Severity.NONE,
    categories: tuple[str, ...] = (),
    groups: tuple[str, ...] = (),
    probe_used: bool = False,
    error: str | None = None,
) -> BiasSignal:
    findings = [
        PolicyViolation(
            policy_id=f"bias_{cat}", category=cat, severity=severity, reason="test finding"
        )
        for cat in categories
    ]
    return BiasSignal(
        status=status,
        score=score,
        severity=severity,
        findings=findings,
        groups_implicated=list(groups),
        probe_used=probe_used,
        error=error,
    )


def cost_signal(
    *,
    status: SignalStatus = SignalStatus.PASS,
    score: float = 0.0,
    total_tokens: int | None = 500,
    latency_ms: float = 800.0,
    over_token_budget: bool = False,
    over_latency_budget: bool = False,
    error: str | None = None,
) -> CostSignal:
    return CostSignal(
        status=status,
        score=score,
        input_tokens=None if total_tokens is None else total_tokens // 2,
        output_tokens=None if total_tokens is None else total_tokens - total_tokens // 2,
        total_tokens=total_tokens,
        estimated_cost_usd=None if total_tokens is None else 0.001,
        llm_latency_ms=latency_ms,
        over_token_budget=over_token_budget,
        over_latency_budget=over_latency_budget,
        error=error,
    )


def assessment(
    *,
    grounding: GroundingSignal | None = None,
    pii: PIISignal | None = None,
    safety: SafetySignal | None = None,
    bias: BiasSignal | None = None,
    cost: CostSignal | None = None,
    weights: dict[str, float] | None = None,
) -> RiskAssessment:
    """Score a signal set through the real scorer, so tests never hand-roll an assessment."""
    signals = RiskSignals(
        grounding=grounding or grounding_signal(),
        pii=pii or pii_signal(),
        safety=safety or safety_signal(),
        bias=bias or bias_signal(),
        cost=cost or cost_signal(),
    )
    return _scorer.score(signals, weights)


__all__ = [
    "assessment",
    "bias_signal",
    "cost_signal",
    "grounding_signal",
    "pii_signal",
    "safety_signal",
]
