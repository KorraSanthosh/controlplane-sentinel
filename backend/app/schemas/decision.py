"""Decision contracts — what ControlPlane decided, and why."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Decision(str, Enum):
    """The tiered action set (FR-09).

    ALLOW  — deliver the model response unchanged.
    REDACT — deliver a modified response (PII masked).
    FLAG   — deliver, but mark for human review; used when evidence is insufficient
             rather than when risk is proven. This is the "do not invent certainty" path.
    BLOCK  — withhold the response, deliver a safe fallback.
    """

    ALLOW = "ALLOW"
    REDACT = "REDACT"
    FLAG = "FLAG"
    BLOCK = "BLOCK"


#: Ordered least → most restrictive. Used when several rules match and the engine is
#: configured to escalate rather than take the first match.
DECISION_SEVERITY = {
    Decision.ALLOW: 0,
    Decision.REDACT: 1,
    Decision.FLAG: 2,
    Decision.BLOCK: 3,
}


class FiredRule(BaseModel):
    """A policy rule that matched, recorded so the audit trail can answer
    "which policy triggered?" (PROJECT_CONTEXT Principle 3).

    ``matched`` holds, per condition, the operator, the expected operand and the *actual*
    observed value. That is what makes a decision re-readable months later: not just "rule X
    fired" but the concrete comparison that made it fire.
    """

    rule_id: str
    action: Decision
    reason: str
    matched: dict[str, Any] = Field(default_factory=dict)


class DecisionResult(BaseModel):
    decision: Decision
    reason: str
    fired_rules: list[FiredRule] = Field(default_factory=list)
    rules_evaluated: int = 0
    policy_profile: str = "default"
    policy_version: str = "0"
    requires_human_review: bool = False
    #: Redaction is orthogonal to the decision tier, not a synonym for ``Decision.REDACT``.
    #: A response can carry PII *and* an unverified claim: the tier then escalates to FLAG
    #: (human review), but the PII must still be masked before delivery. Collapsing the two
    #: would silently deliver PII whenever a higher tier won the tie.
    apply_redaction: bool = False


__all__ = [
    "DECISION_SEVERITY",
    "Decision",
    "DecisionResult",
    "FiredRule",
]
