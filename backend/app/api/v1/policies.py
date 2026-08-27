"""``/policies`` — read the active governance configuration.

Read-only by design. Editing a policy through the API would mean a decision could be made under
one configuration and re-read under another with no record of the change; the profile's
``version`` and ``source_path`` are what make an audit record re-readable months later. Policy
changes belong in the YAML files, under version control.

The response deliberately exposes the thresholds and weights verbatim, including the reminder
that they are prototype configuration rather than industry standards.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import PoliciesDep
from app.schemas.decision import Decision
from app.services.policy.loader import PolicyProfile

router = APIRouter(tags=["policies"])

THRESHOLD_DISCLAIMER = (
    "All weights, budgets and thresholds below are prototype policy configuration chosen for "
    "this demonstration. They are not industry standards, benchmarks or calibrated values."
)


class ConditionView(BaseModel):
    field: str
    operator: str
    operand: object


class RuleView(BaseModel):
    id: str
    action: Decision
    reason: str
    requires_human_review: bool
    apply_redaction: bool
    when: list[ConditionView]


class BudgetView(BaseModel):
    max_total_tokens: int
    max_latency_ms: float


class TriageView(BaseModel):
    deep_grounding_enabled: bool
    deep_grounding_on_claims: bool
    judge_enabled: bool
    judge_on_fast_risk_at_or_above: float
    judge_on_grounding_status: list[str]
    judge_on_safety_status: list[str]


class ProfileView(BaseModel):
    id: str
    version: str
    title: str
    description: str
    disclaimer: str = THRESHOLD_DISCLAIMER
    enabled_checks: dict[str, bool]
    weights: dict[str, float]
    default_budget: BudgetView
    budgets: dict[str, BudgetView]
    triage: TriageView
    on_unavailable: dict[str, Decision]
    strategy: str
    default_action: Decision
    default_reason: str
    blocked_response: str
    safety_categories: list[str] | None = None
    rules: list[RuleView] = Field(default_factory=list)

    @classmethod
    def of(cls, profile: PolicyProfile) -> "ProfileView":
        return cls(
            id=profile.id,
            version=profile.version,
            title=profile.title,
            description=profile.description,
            enabled_checks=profile.enabled_checks,
            weights=profile.weights,
            default_budget=BudgetView(
                max_total_tokens=profile.default_budget.max_total_tokens,
                max_latency_ms=profile.default_budget.max_latency_ms,
            ),
            budgets={
                name: BudgetView(
                    max_total_tokens=b.max_total_tokens, max_latency_ms=b.max_latency_ms
                )
                for name, b in profile.budgets.items()
            },
            triage=TriageView(
                deep_grounding_enabled=profile.triage.deep_grounding_enabled,
                deep_grounding_on_claims=profile.triage.deep_grounding_on_claims,
                judge_enabled=profile.triage.judge_enabled,
                judge_on_fast_risk_at_or_above=profile.triage.judge_on_fast_risk_at_or_above,
                judge_on_grounding_status=sorted(
                    s.value for s in profile.triage.judge_on_grounding_status
                ),
                judge_on_safety_status=sorted(
                    s.value for s in profile.triage.judge_on_safety_status
                ),
            ),
            on_unavailable=profile.on_unavailable,
            strategy=profile.strategy,
            default_action=profile.default_action,
            default_reason=profile.default_reason,
            blocked_response=profile.blocked_response,
            safety_categories=list(profile.safety_categories)
            if profile.safety_categories
            else None,
            rules=[
                RuleView(
                    id=rule.id,
                    action=rule.action,
                    reason=rule.reason,
                    requires_human_review=rule.requires_human_review,
                    apply_redaction=rule.apply_redaction,
                    when=[
                        ConditionView(
                            field=c.field, operator=c.operator, operand=c.operand
                        )
                        for c in rule.conditions
                    ],
                )
                for rule in profile.rules
            ],
        )


class PolicyList(BaseModel):
    default_profile: str
    disclaimer: str = THRESHOLD_DISCLAIMER
    profiles: list[ProfileView]


@router.get("/policies", response_model=PolicyList, summary="All loaded policy profiles")
async def list_policies(policies: PoliciesDep) -> PolicyList:
    return PolicyList(
        default_profile=policies.default_id,
        profiles=[ProfileView.of(policies.profiles[pid]) for pid in policies.ids()],
    )


@router.get(
    "/policies/{profile_id}", response_model=ProfileView, summary="One policy profile"
)
async def get_policy(profile_id: str, policies: PoliciesDep) -> ProfileView:
    profile = policies.profiles.get(profile_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No policy profile '{profile_id}'. Loaded profiles: "
                f"{', '.join(policies.ids())}."
            ),
        )
    return ProfileView.of(profile)


__all__ = ["PolicyList", "ProfileView", "THRESHOLD_DISCLAIMER", "router"]
