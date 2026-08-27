"""Public API contracts for the /chat endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.decision import Decision, FiredRule
from app.schemas.signals import RiskAssessment, Telemetry


class ChatRequest(BaseModel):
    """FR-01 request intake."""

    message: str = Field(min_length=1, max_length=8000)
    use_case: str = "support_assistant"
    policy_profile: str | None = None  # None → settings.default_policy_profile
    user_id: str | None = None
    tenant_id: str | None = None
    #: Reveal the pre-action model text alongside the delivered text. Powers the
    #: dashboard's "original vs delivered" view. Honoured only when the server has
    #: CP_ALLOW_DEBUG_ORIGINAL enabled — otherwise ignored, because returning text the
    #: policy engine chose to withhold would defeat the entire control layer.
    debug: bool = False


class ChatResponse(BaseModel):
    """What the caller receives.

    ``answer`` is always the *delivered* text: unchanged on ALLOW/FLAG, masked on REDACT,
    a safe fallback message on BLOCK. ``original_answer`` is populated only in debug mode.
    """

    request_id: str
    answer: str
    decision: Decision
    reason: str
    original_answer: str | None = None
    answer_modified: bool = False
    original_withheld: bool = False
    requires_human_review: bool = False
    risk: RiskAssessment
    telemetry: Telemetry
    fired_rules: list[FiredRule] = Field(default_factory=list)
    provider: str = ""
    model: str = ""
    policy_profile: str = "default"


__all__ = ["ChatRequest", "ChatResponse"]
