"""Risk signal contracts.

Every detector returns a subclass of :class:`BaseSignal`. Two rules hold everywhere:

1. ``score`` is *risk*, normalised 0.0 (no risk) .. 1.0 (maximum risk). Never a
   confidence-in-the-answer score — mixing the two directions is a classic bug source.
2. ``status`` always includes ``UNAVAILABLE``. A detector that crashes or whose dependency
   is down MUST report ``UNAVAILABLE`` and never ``PASS`` (SYSTEM_REQUIREMENTS FR-11).
   ``SKIPPED`` is different: it means triage deliberately did not run this check.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SignalStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    UNAVAILABLE = "unavailable"  # could not run — never conflate with PASS
    SKIPPED = "skipped"  # deliberately not run by triage/policy


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_SEVERITY_ORDER = {
    Severity.NONE: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


def severity_rank(sev: Severity) -> int:
    """Numeric rank so policies can express `severity >= high`."""
    return _SEVERITY_ORDER[sev]


class BaseSignal(BaseModel):
    """Common shape shared by every detector result."""

    name: str
    status: SignalStatus
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Risk, 0=none 1=max")
    severity: Severity = Severity.NONE
    explanation: str = ""
    error: str | None = None
    duration_ms: float = 0.0

    @property
    def usable(self) -> bool:
        """True when this signal's score can be trusted in aggregation."""
        return self.status not in (SignalStatus.UNAVAILABLE, SignalStatus.SKIPPED)


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------
class GroundingStatus(str, Enum):
    """FR-04 mandates exactly these four states."""

    GROUNDED = "grounded"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNAVAILABLE = "unavailable"


class Evidence(BaseModel):
    """One trusted fact retrieved from the graph, and how it relates to a claim.

    ``supports=False`` means the fact *refutes* the claim — that is what turns a
    hallucination into a CONTRADICTED status rather than merely UNSUPPORTED.
    """

    subject: str
    predicate: str
    object: str
    reference: str  # human-readable, e.g. "PlanPremium-INCLUDES_ROAMING-false"
    supports: bool
    source_document: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Claim(BaseModel):
    """A single factual assertion extracted from the model response."""

    text: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    status: GroundingStatus = GroundingStatus.UNSUPPORTED
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)
    explanation: str = ""


class GroundingSignal(BaseSignal):
    name: Literal["grounding"] = "grounding"
    grounding_status: GroundingStatus = GroundingStatus.UNAVAILABLE
    claims: list[Claim] = Field(default_factory=list)
    claims_checked: int = 0
    claims_grounded: int = 0
    claims_unsupported: int = 0
    claims_contradicted: int = 0
    #: Assertions the graph holds no fact about for that subject+predicate. Counted and
    #: surfaced, but excluded from the status roll-up: a closed-world graph cannot honestly
    #: call an assertion false merely because it is outside the graph's coverage. This is
    #: the system's main known false-negative channel and is documented as such.
    claims_unverifiable: int = 0
    graph_backend: str = "unknown"  # "neo4j" | "memory" | "unavailable"
    #: True when at least one contradicted claim carries refuting evidence with a concrete
    #: object value — i.e. the graph does not merely disagree, it supplies the correct value.
    #: This is what separates a response that can be *repaired* from one that can only be
    #: withheld, and it is a policy-visible field (``grounding.repairable``) for that reason.
    repairable: bool = False


# ---------------------------------------------------------------------------
# PII
# ---------------------------------------------------------------------------
class PIIKind(str, Enum):
    EMAIL = "email"
    PHONE = "phone"
    ACCOUNT_NUMBER = "account_number"
    CREDIT_CARD = "credit_card"
    NATIONAL_ID = "national_id"
    IP_ADDRESS = "ip_address"
    POSTAL_ADDRESS = "postal_address"
    DATE_OF_BIRTH = "date_of_birth"


class PIIMatch(BaseModel):
    """A detected PII span.

    ``start``/``end`` are character offsets used by the redactor at request time.
    ``preview`` is already masked — the raw matched value is never carried on this model,
    so an audit record built from it cannot leak the original (NFR-01).
    """

    kind: PIIKind
    start: int
    end: int
    preview: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class PIISignal(BaseSignal):
    name: Literal["pii"] = "pii"
    detected: bool = False
    redactable: bool = False
    matches: list[PIIMatch] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Safety / policy
# ---------------------------------------------------------------------------
class PolicyViolation(BaseModel):
    policy_id: str
    category: str
    severity: Severity
    reason: str
    matched_preview: str = ""


class SafetySignal(BaseSignal):
    name: Literal["safety"] = "safety"
    violations: list[PolicyViolation] = Field(default_factory=list)
    judge_used: bool = False  # True when the deep-path LLM judge ran
    judge_verdict: str | None = None


# ---------------------------------------------------------------------------
# Bias / fairness
# ---------------------------------------------------------------------------
class BiasSignal(BaseSignal):
    """Fairness risk: does the response treat people differently by group?

    Reuses :class:`PolicyViolation` for findings rather than defining a near-identical model,
    so ``policy_id`` / ``category`` / ``severity`` mean the same thing here as in safety and
    the audit trail stays uniform.

    ``groups_implicated`` records *which* protected attribute a finding turned on (age, gender,
    ethnicity, a postcode used as an income proxy). That is the part a reviewer needs, and a
    single aggregate score cannot carry it.
    """

    name: Literal["bias"] = "bias"
    findings: list[PolicyViolation] = Field(default_factory=list)
    groups_implicated: list[str] = Field(default_factory=list)
    #: True when the deep-path LLM probe ran. The rule layer stays authoritative for status,
    #: exactly as with safety.
    probe_used: bool = False
    probe_verdict: str | None = None


# ---------------------------------------------------------------------------
# Cost / latency
# ---------------------------------------------------------------------------
class CostSignal(BaseSignal):
    name: Literal["cost"] = "cost"
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    llm_latency_ms: float = 0.0
    over_token_budget: bool = False
    over_latency_budget: bool = False


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
class RiskSignals(BaseModel):
    """The full signal set for one request."""

    grounding: GroundingSignal
    pii: PIISignal
    safety: SafetySignal
    bias: BiasSignal
    cost: CostSignal

    def as_dict(self) -> dict[str, BaseSignal]:
        return {
            "grounding": self.grounding,
            "pii": self.pii,
            "safety": self.safety,
            "bias": self.bias,
            "cost": self.cost,
        }


class RiskAssessment(BaseModel):
    """Component scores plus the aggregated total.

    Both are persisted (SYSTEM_REQUIREMENTS §8): a single number is not explainable on
    its own, and the weights used are recorded so a past decision can be re-read against
    the policy version that produced it.
    """

    overall_score: float = Field(ge=0.0, le=1.0)
    component_scores: dict[str, float] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    unavailable_checks: list[str] = Field(default_factory=list)
    skipped_checks: list[str] = Field(default_factory=list)
    signals: RiskSignals
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
class StageTiming(BaseModel):
    stage: str
    duration_ms: float


class Telemetry(BaseModel):
    """Latency accounting.

    ``controlplane_overhead_ms`` is deliberately separate from ``llm_latency_ms``: the
    brief asks how the checker protects latency, and the honest answer is a measured
    overhead figure plus the fraction of traffic that took the deep path.
    """

    llm_latency_ms: float = 0.0
    controlplane_overhead_ms: float = 0.0
    total_ms: float = 0.0
    path_taken: Literal["fast", "deep"] = "fast"
    deep_path_reasons: list[str] = Field(default_factory=list)
    stages: list[StageTiming] = Field(default_factory=list)
    #: True when the judge was reached only because grounding came back contradicted or
    #: unsupported. That route runs *after* the graph query rather than alongside it, so it
    #: costs an extra sequential round trip. Surfaced separately because ``path_taken`` alone
    #: would fold the worst case in with the cheaper concurrent one.
    judge_escalated_after_grounding: bool = False
    #: True when ``llm_latency_ms`` came from a scripted scenario rather than a real network
    #: call. Surfaced so a demo can show a 9.6-second response without the timing numbers
    #: quietly implying that wall-clock was measured. ``controlplane_overhead_ms`` is always
    #: really measured, whichever provider ran.
    llm_latency_simulated: bool = False


__all__ = [
    "BaseSignal",
    "BiasSignal",
    "Claim",
    "CostSignal",
    "Evidence",
    "GroundingSignal",
    "GroundingStatus",
    "PIIKind",
    "PIIMatch",
    "PIISignal",
    "PolicyViolation",
    "RiskAssessment",
    "RiskSignals",
    "SafetySignal",
    "Severity",
    "SignalStatus",
    "StageTiming",
    "Telemetry",
    "severity_rank",
]
