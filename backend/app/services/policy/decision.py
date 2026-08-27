"""The decision engine (FR-09).

Maps a :class:`RiskAssessment` onto one of ALLOW / REDACT / FLAG / BLOCK by evaluating the
active profile's rules. Three properties matter more than the mapping itself:

**Fail-safe first.** Before any rule is considered, every detector that reported
``UNAVAILABLE`` is checked against the profile's ``on_unavailable`` map. A crashed grounding
checker cannot produce a contradiction, so no content rule would ever fire for it — without
this pass, a broken detector would read exactly like a clean response. This is FR-11 enforced
at the decision layer rather than trusted to rule authors.

**Redaction is orthogonal to the tier.** A response can carry both PII and an unverified claim.
The tier escalates to FLAG because a human must look at it, but the PII still has to be masked
before delivery. So ``apply_redaction`` is tracked separately from which rule won.

**Every decision explains itself.** The winning rule, all matching rules, and the actual
observed value behind each matched condition are recorded. Nothing about a decision is implicit.
"""

from __future__ import annotations

import logging

from app.schemas.decision import DECISION_SEVERITY, Decision, DecisionResult, FiredRule
from app.schemas.signals import RiskAssessment, SignalStatus
from app.services.policy.loader import (
    STRATEGY_FIRST_MATCH,
    DecisionRule,
    PolicyProfile,
)

logger = logging.getLogger(__name__)

#: Decisions that mean "a person must look at this".
_REVIEW_DECISIONS = frozenset({Decision.FLAG})


class DecisionEngine:
    name = "decision"

    def decide(self, assessment: RiskAssessment, profile: PolicyProfile) -> DecisionResult:
        fired: list[FiredRule] = []

        fired.extend(self._failsafe_rules(assessment, profile))
        content_fired, evaluated = self._match_rules(assessment, profile)
        fired.extend(content_fired)

        decision, reason, winner = self._resolve(fired, profile)
        matched_rules = content_fired_rules(content_fired, profile)

        # Redaction follows the PII rule that fired, not the tier that won — see module
        # docstring. Nothing is delivered on BLOCK, so masking is moot there.
        apply_redaction = decision is not Decision.BLOCK and any(
            rule.apply_redaction for rule in matched_rules
        )

        requires_review = decision in _REVIEW_DECISIONS or any(
            rule.requires_human_review for rule in matched_rules
        )

        # A fail-safe that escalated the tier always wants human eyes: the system is acting on
        # missing information, which is precisely the case it should not resolve alone.
        if winner is not None and winner.rule_id.startswith("failsafe."):
            requires_review = requires_review or decision is not Decision.ALLOW

        return DecisionResult(
            decision=decision,
            reason=reason,
            fired_rules=fired,
            rules_evaluated=evaluated,
            policy_profile=profile.id,
            policy_version=profile.version,
            requires_human_review=requires_review,
            apply_redaction=apply_redaction,
        )

    # -- fail-safe ---------------------------------------------------------
    def _failsafe_rules(
        self, assessment: RiskAssessment, profile: PolicyProfile
    ) -> list[FiredRule]:
        fired: list[FiredRule] = []
        for name, signal in assessment.signals.as_dict().items():
            if signal.status is not SignalStatus.UNAVAILABLE:
                continue
            action = profile.unavailable_action(name)
            fired.append(
                FiredRule(
                    rule_id=f"failsafe.{name}_unavailable",
                    action=action,
                    reason=(
                        f"The {name} check could not run, so its result is unknown rather than "
                        f"clean. Policy '{profile.id}' handles an unavailable {name} check with "
                        f"{action.value}."
                    ),
                    matched={
                        f"{name}.signal_status": {
                            "operator": "eq",
                            "expected": SignalStatus.UNAVAILABLE.value,
                            "actual": signal.status.value,
                        },
                        "detector_error": signal.error or "not reported",
                    },
                )
            )
        return fired

    # -- content rules -----------------------------------------------------
    def _match_rules(
        self, assessment: RiskAssessment, profile: PolicyProfile
    ) -> tuple[list[FiredRule], int]:
        fired: list[FiredRule] = []
        evaluated = 0

        for rule in profile.rules:
            evaluated += 1
            matched = True
            trace: dict[str, object] = {}

            # Conditions are ANDed. Every condition is evaluated even after one fails, because
            # the trace of what the values actually were is useful on near-miss rules too.
            for condition in rule.conditions:
                ok, detail = condition.evaluate(assessment)
                trace[condition.field] = detail
                matched = matched and ok

            if matched:
                fired.append(
                    FiredRule(
                        rule_id=rule.id, action=rule.action, reason=rule.reason, matched=trace
                    )
                )

            if matched and profile.strategy == STRATEGY_FIRST_MATCH:
                break

        return fired, evaluated

    # -- resolution --------------------------------------------------------
    def _resolve(
        self, fired: list[FiredRule], profile: PolicyProfile
    ) -> tuple[Decision, str, FiredRule | None]:
        if not fired:
            return profile.default_action, profile.default_reason, None

        if profile.strategy == STRATEGY_FIRST_MATCH:
            winner = fired[0]
        else:
            # Harshest action wins; ties break toward the earliest fired rule, so the reason
            # text is stable and follows file order rather than dict iteration luck.
            harshest = max(DECISION_SEVERITY[r.action] for r in fired)
            winner = next(r for r in fired if DECISION_SEVERITY[r.action] == harshest)

        if winner.action is Decision.ALLOW:
            # Only advisory rules matched (a cost warning, typically). The action is still
            # ALLOW, but the rule's own reason is more informative than the generic default.
            return Decision.ALLOW, winner.reason, winner

        return winner.action, winner.reason, winner


def content_fired_rules(
    fired: list[FiredRule], profile: PolicyProfile
) -> list[DecisionRule]:
    """Look the profile's rule objects back up from the fired records.

    ``FiredRule`` is the serialisable audit shape and deliberately does not carry the
    behavioural flags; the engine needs the originals to read ``apply_redaction`` and
    ``requires_human_review``.
    """
    by_id = {rule.id: rule for rule in profile.rules}
    return [by_id[f.rule_id] for f in fired if f.rule_id in by_id]


__all__ = ["DecisionEngine"]
