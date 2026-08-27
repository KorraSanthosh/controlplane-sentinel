"""Policy loading, condition validation and the decision engine.

The load-time validation tests matter as much as the decision tests. A rule with a mistyped
field name that silently never matches is indistinguishable from a rule that always passes, and
the second one is a governance hole — so the loader has to reject it at startup.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.schemas.decision import Decision
from app.schemas.signals import GroundingStatus, PIIKind, Severity, SignalStatus
from app.services.policy.conditions import PolicyError, build_conditions
from app.services.policy.decision import DecisionEngine
from app.services.policy.loader import (
    STRATEGY_FIRST_MATCH,
    PolicyProfile,
    PolicyRegistry,
    load_policy_registry,
    load_profile,
)
from tests.factories import (
    assessment,
    cost_signal,
    grounding_signal,
    pii_signal,
    safety_signal,
)


@pytest.fixture
def engine() -> DecisionEngine:
    return DecisionEngine()


@pytest.fixture
def default(policies: PolicyRegistry) -> PolicyProfile:
    return policies.get("default")


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
def test_three_profiles_load(policies: PolicyRegistry) -> None:
    assert policies.ids() == ["default", "lenient", "strict"]
    assert policies.default_id == "default"


def test_every_profile_is_internally_consistent(policies: PolicyRegistry) -> None:
    for profile in policies.profiles.values():
        assert profile.version, f"{profile.id} has no version — audits could not be re-read"
        assert profile.rules, f"{profile.id} has no rules"
        assert sum(profile.weights.values()) > 0
        assert profile.default_budget.max_total_tokens > 0
        for rule in profile.rules:
            assert rule.conditions, f"{profile.id}.{rule.id} matches everything"
            assert rule.reason, f"{profile.id}.{rule.id} is unexplained"


def test_unknown_profile_falls_back_rather_than_raising(policies: PolicyRegistry) -> None:
    """A bad profile id in a request body must not 500, and must not run no policy at all."""
    assert policies.get("does-not-exist").id == "default"


def test_safety_rules_file_is_not_loaded_as_a_profile(policies: PolicyRegistry) -> None:
    assert "safety_rules" not in policies.profiles


def _write_profile(tmp_path: Path, **overrides) -> Path:
    body = {
        "id": "probe",
        "version": "1",
        "weights": {"safety": 1.0},
        "budgets": {"default": {"max_total_tokens": 100, "max_latency_ms": 100}},
        "rules": [{"id": "r", "action": "BLOCK", "reason": "because", "when": {"pii.detected": True}}],
    }
    body.update(overrides)
    path = tmp_path / "probe.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def test_unknown_condition_field_is_rejected_at_load_time(tmp_path: Path) -> None:
    path = _write_profile(
        tmp_path,
        rules=[{"id": "r", "action": "BLOCK", "reason": "x", "when": {"pii.detcted": True}}],
    )
    with pytest.raises(PolicyError, match="unknown condition field"):
        load_profile(path)


def test_operator_illegal_for_the_field_type_is_rejected(tmp_path: Path) -> None:
    """`status: {gte: warn}` would compare enum declaration order by accident."""
    path = _write_profile(
        tmp_path,
        rules=[
            {
                "id": "r",
                "action": "BLOCK",
                "reason": "x",
                "when": {"safety.status": {"gte": "warn"}},
            }
        ],
    )
    with pytest.raises(PolicyError, match="not valid for a signal_status field"):
        load_profile(path)


def test_rule_without_conditions_is_rejected(tmp_path: Path) -> None:
    path = _write_profile(
        tmp_path, rules=[{"id": "r", "action": "BLOCK", "reason": "x", "when": {}}]
    )
    with pytest.raises(PolicyError, match="at least one condition"):
        load_profile(path)


def test_rule_without_reason_is_rejected(tmp_path: Path) -> None:
    path = _write_profile(
        tmp_path, rules=[{"id": "r", "action": "BLOCK", "when": {"pii.detected": True}}]
    )
    with pytest.raises(PolicyError, match="must be explainable"):
        load_profile(path)


def test_duplicate_rule_ids_are_rejected(tmp_path: Path) -> None:
    rule = {"id": "same", "action": "BLOCK", "reason": "x", "when": {"pii.detected": True}}
    path = _write_profile(tmp_path, rules=[rule, dict(rule)])
    with pytest.raises(PolicyError, match="duplicate rule id"):
        load_profile(path)


def test_unknown_action_is_rejected(tmp_path: Path) -> None:
    path = _write_profile(
        tmp_path,
        rules=[{"id": "r", "action": "DESTROY", "reason": "x", "when": {"pii.detected": True}}],
    )
    with pytest.raises(PolicyError, match="is not a decision"):
        load_profile(path)


def test_unknown_weight_name_is_rejected(tmp_path: Path) -> None:
    path = _write_profile(tmp_path, weights={"grounding": 0.5, "vibes": 0.5})
    with pytest.raises(PolicyError, match="unknown check 'vibes'"):
        load_profile(path)


def test_missing_default_budget_is_rejected(tmp_path: Path) -> None:
    path = _write_profile(tmp_path, budgets={"support": {"max_total_tokens": 1, "max_latency_ms": 1}})
    with pytest.raises(PolicyError, match="a 'default' budget is required"):
        load_profile(path)


def test_unknown_strategy_is_rejected(tmp_path: Path) -> None:
    path = _write_profile(tmp_path, strategy="whatever_feels_right")
    with pytest.raises(PolicyError, match="is not valid"):
        load_profile(path)


def test_membership_operator_requires_a_list() -> None:
    with pytest.raises(PolicyError, match="needs a list"):
        build_conditions({"grounding.status": {"in": "contradicted"}}, "probe")


def test_missing_default_profile_is_a_startup_failure(tmp_path: Path) -> None:
    _write_profile(tmp_path)
    with pytest.raises(PolicyError, match="Default policy profile 'default' not found"):
        load_policy_registry(tmp_path, "default")


def test_empty_policy_directory_is_a_startup_failure(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match="No policy profiles found"):
        load_policy_registry(tmp_path, "default")


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------
def test_clean_response_is_allowed(engine: DecisionEngine, default: PolicyProfile) -> None:
    result = engine.decide(assessment(), default)
    assert result.decision is Decision.ALLOW
    assert result.fired_rules == []
    assert result.reason == default.default_reason
    assert not result.requires_human_review


def test_critical_safety_blocks(engine: DecisionEngine, default: PolicyProfile) -> None:
    result = engine.decide(
        assessment(
            safety=safety_signal(
                status=SignalStatus.FAIL,
                score=1.0,
                severity=Severity.CRITICAL,
                categories=("unauthorised_commitment",),
            )
        ),
        default,
    )
    assert result.decision is Decision.BLOCK
    assert "critical_safety" in [r.rule_id for r in result.fired_rules]


def test_contradicted_fact_blocks(engine: DecisionEngine, default: PolicyProfile) -> None:
    result = engine.decide(
        assessment(
            grounding=grounding_signal(
                status=SignalStatus.FAIL,
                grounding_status=GroundingStatus.CONTRADICTED,
                score=0.95,
                severity=Severity.CRITICAL,
                claims_checked=1,
                claims_contradicted=1,
            )
        ),
        default,
    )
    assert result.decision is Decision.BLOCK
    assert "contradicted_fact" in [r.rule_id for r in result.fired_rules]


def test_unsupported_claim_flags_for_review(
    engine: DecisionEngine, default: PolicyProfile
) -> None:
    result = engine.decide(
        assessment(
            grounding=grounding_signal(
                status=SignalStatus.WARN,
                grounding_status=GroundingStatus.UNSUPPORTED,
                score=0.55,
                severity=Severity.MEDIUM,
                claims_checked=1,
                claims_unsupported=1,
            )
        ),
        default,
    )
    assert result.decision is Decision.FLAG
    assert result.requires_human_review


def test_detectable_pii_redacts(engine: DecisionEngine, default: PolicyProfile) -> None:
    result = engine.decide(
        assessment(
            pii=pii_signal(
                status=SignalStatus.FAIL,
                score=0.6,
                severity=Severity.MEDIUM,
                detected=True,
                redactable=True,
                kinds=(PIIKind.EMAIL,),
            )
        ),
        default,
    )
    assert result.decision is Decision.REDACT
    assert result.apply_redaction


def test_redaction_survives_escalation_to_flag(
    engine: DecisionEngine, default: PolicyProfile
) -> None:
    """A FLAG that carries PII is still delivered, so the PII must still be masked."""
    result = engine.decide(
        assessment(
            grounding=grounding_signal(
                status=SignalStatus.WARN,
                grounding_status=GroundingStatus.UNSUPPORTED,
                score=0.55,
                severity=Severity.MEDIUM,
                claims_checked=1,
                claims_unsupported=1,
            ),
            pii=pii_signal(
                status=SignalStatus.FAIL,
                score=0.6,
                severity=Severity.MEDIUM,
                detected=True,
                redactable=True,
                kinds=(PIIKind.EMAIL,),
            ),
        ),
        default,
    )
    assert result.decision is Decision.FLAG
    assert result.apply_redaction, "masking must not be dropped because a harsher tier won"


def test_redaction_is_moot_on_block(engine: DecisionEngine, default: PolicyProfile) -> None:
    result = engine.decide(
        assessment(
            safety=safety_signal(
                status=SignalStatus.FAIL, score=1.0, severity=Severity.CRITICAL
            ),
            pii=pii_signal(
                status=SignalStatus.FAIL,
                score=0.6,
                detected=True,
                redactable=True,
                kinds=(PIIKind.EMAIL,),
            ),
        ),
        default,
    )
    assert result.decision is Decision.BLOCK
    assert not result.apply_redaction


def test_cost_overrun_alone_does_not_change_the_decision(
    engine: DecisionEngine, default: PolicyProfile
) -> None:
    result = engine.decide(
        assessment(
            cost=cost_signal(
                status=SignalStatus.WARN,
                score=1.0,
                total_tokens=9000,
                latency_ms=12000.0,
                over_token_budget=True,
                over_latency_budget=True,
            )
        ),
        default,
    )
    assert result.decision is Decision.ALLOW
    assert "cost_anomaly" in [r.rule_id for r in result.fired_rules]
    assert "budget" in result.reason.lower()


def test_most_restrictive_wins_regardless_of_file_order(
    engine: DecisionEngine, default: PolicyProfile
) -> None:
    result = engine.decide(
        assessment(
            grounding=grounding_signal(
                status=SignalStatus.FAIL,
                grounding_status=GroundingStatus.CONTRADICTED,
                score=0.95,
                severity=Severity.CRITICAL,
                claims_checked=1,
                claims_contradicted=1,
            ),
            pii=pii_signal(
                status=SignalStatus.FAIL,
                score=0.6,
                detected=True,
                redactable=True,
                kinds=(PIIKind.EMAIL,),
            ),
            cost=cost_signal(status=SignalStatus.WARN, score=1.0, over_token_budget=True),
        ),
        default,
    )
    fired = [r.rule_id for r in result.fired_rules]
    assert {"contradicted_fact", "pii_redactable", "cost_anomaly"} <= set(fired)
    assert result.decision is Decision.BLOCK


def test_first_match_stops_at_the_first_rule(engine: DecisionEngine, tmp_path: Path) -> None:
    path = _write_profile(
        tmp_path,
        strategy=STRATEGY_FIRST_MATCH,
        rules=[
            {
                "id": "redact_first",
                "action": "REDACT",
                "reason": "mask it",
                "when": {"pii.detected": True},
            },
            {
                "id": "block_later",
                "action": "BLOCK",
                "reason": "withhold it",
                "when": {"pii.detected": True},
            },
        ],
    )
    profile = load_profile(path)
    result = engine.decide(
        assessment(
            pii=pii_signal(
                status=SignalStatus.FAIL, score=0.6, detected=True, redactable=True,
                kinds=(PIIKind.EMAIL,),
            )
        ),
        profile,
    )
    assert result.decision is Decision.REDACT
    assert [r.rule_id for r in result.fired_rules] == ["redact_first"]


def test_every_fired_rule_records_what_it_observed(
    engine: DecisionEngine, default: PolicyProfile
) -> None:
    result = engine.decide(
        assessment(
            safety=safety_signal(
                status=SignalStatus.FAIL, score=1.0, severity=Severity.CRITICAL
            )
        ),
        default,
    )
    fired = next(r for r in result.fired_rules if r.rule_id == "critical_safety")
    trace = fired.matched["safety.severity"]
    assert trace["operator"] == "gte"
    assert trace["expected"] == "critical"
    assert trace["actual"] == "critical"


def test_policy_identity_is_recorded_on_the_decision(
    engine: DecisionEngine, default: PolicyProfile
) -> None:
    result = engine.decide(assessment(), default)
    assert result.policy_profile == "default"
    assert result.policy_version == default.version


# ---------------------------------------------------------------------------
# Fail-safe (FR-11)
# ---------------------------------------------------------------------------
def test_unavailable_grounding_flags_not_allows(
    engine: DecisionEngine, default: PolicyProfile
) -> None:
    """The core fail-safe: a detector that could not run is not a clean bill of health."""
    result = engine.decide(
        assessment(
            grounding=grounding_signal(
                status=SignalStatus.UNAVAILABLE,
                grounding_status=GroundingStatus.UNAVAILABLE,
                severity=Severity.MEDIUM,
                error="graph unreachable",
            )
        ),
        default,
    )
    assert result.decision is Decision.FLAG
    assert result.requires_human_review
    failsafe = next(r for r in result.fired_rules if r.rule_id == "failsafe.grounding_unavailable")
    assert failsafe.matched["detector_error"] == "graph unreachable"


def test_unavailable_cost_is_allowed_because_the_profile_says_so(
    engine: DecisionEngine, default: PolicyProfile
) -> None:
    """Missing token counts are an accounting gap, not a user-facing risk — but it is explicit."""
    result = engine.decide(
        assessment(cost=cost_signal(status=SignalStatus.UNAVAILABLE, total_tokens=None)),
        default,
    )
    assert result.decision is Decision.ALLOW
    assert "failsafe.cost_unavailable" in [r.rule_id for r in result.fired_rules]


def test_unspecified_detector_defaults_to_flag(tmp_path: Path, engine: DecisionEngine) -> None:
    """A profile that forgets a detector must not thereby allow it through."""
    path = _write_profile(tmp_path, on_unavailable={})
    profile = load_profile(path)
    assert profile.unavailable_action("grounding") is Decision.FLAG
    result = engine.decide(
        assessment(
            grounding=grounding_signal(
                status=SignalStatus.UNAVAILABLE, grounding_status=GroundingStatus.UNAVAILABLE
            )
        ),
        profile,
    )
    assert result.decision is Decision.FLAG


def test_skipped_detector_does_not_trip_the_fail_safe(
    engine: DecisionEngine, default: PolicyProfile
) -> None:
    """SKIPPED is a deliberate triage decision, not a failure."""
    result = engine.decide(
        assessment(grounding=grounding_signal(status=SignalStatus.SKIPPED)), default
    )
    assert result.decision is Decision.ALLOW
    assert not any(r.rule_id.startswith("failsafe.") for r in result.fired_rules)


def test_strict_profile_is_harsher_than_lenient_on_the_same_signals(
    engine: DecisionEngine, policies: PolicyRegistry
) -> None:
    """Posture is configuration: identical evidence, different action."""
    unsupported = assessment(
        grounding=grounding_signal(
            status=SignalStatus.WARN,
            grounding_status=GroundingStatus.UNSUPPORTED,
            score=0.55,
            severity=Severity.MEDIUM,
            claims_checked=1,
            claims_unsupported=1,
        )
    )
    strict = engine.decide(unsupported, policies.get("strict")).decision
    lenient = engine.decide(unsupported, policies.get("lenient")).decision
    assert strict is not lenient
