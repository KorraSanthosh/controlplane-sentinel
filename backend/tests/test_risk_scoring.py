"""Risk aggregation math.

The behaviour worth pinning down is renormalisation. When a detector cannot score, its weight
has to leave the denominator — otherwise the maximum achievable risk drops precisely when the
system knows least, and a genuinely dangerous response scores lower than a mildly risky one.
"""

from __future__ import annotations

import pytest

from app.schemas.signals import (
    BiasSignal,
    CostSignal,
    GroundingSignal,
    GroundingStatus,
    PIISignal,
    RiskSignals,
    SafetySignal,
    Severity,
    SignalStatus,
)
from app.services.risk.scoring import DEFAULT_WEIGHTS, RiskScorer

WEIGHTS = {"grounding": 0.40, "safety": 0.30, "pii": 0.20, "cost": 0.10}


def signals(
    *,
    grounding: tuple[SignalStatus, float] = (SignalStatus.PASS, 0.0),
    pii: tuple[SignalStatus, float] = (SignalStatus.PASS, 0.0),
    safety: tuple[SignalStatus, float] = (SignalStatus.PASS, 0.0),
    bias: tuple[SignalStatus, float] = (SignalStatus.PASS, 0.0),
    cost: tuple[SignalStatus, float] = (SignalStatus.PASS, 0.0),
) -> RiskSignals:
    return RiskSignals(
        grounding=GroundingSignal(
            status=grounding[0],
            score=grounding[1],
            grounding_status=GroundingStatus.GROUNDED,
        ),
        pii=PIISignal(status=pii[0], score=pii[1]),
        safety=SafetySignal(status=safety[0], score=safety[1]),
        bias=BiasSignal(status=bias[0], score=bias[1]),
        cost=CostSignal(status=cost[0], score=cost[1]),
    )


@pytest.fixture
def scorer() -> RiskScorer:
    return RiskScorer()


def test_all_clean_scores_zero(scorer: RiskScorer) -> None:
    result = scorer.score(signals(), WEIGHTS)
    assert result.overall_score == 0.0
    assert result.unavailable_checks == []
    assert result.skipped_checks == []


def test_weighted_sum_over_full_weight_set(scorer: RiskScorer) -> None:
    result = scorer.score(
        signals(
            grounding=(SignalStatus.FAIL, 1.0),
            safety=(SignalStatus.PASS, 0.5),
        ),
        WEIGHTS,
    )
    # (0.40*1.0 + 0.30*0.5 + 0.20*0 + 0.10*0) / 1.00
    assert result.overall_score == pytest.approx(0.55)


def test_unavailable_detector_leaves_the_denominator(scorer: RiskScorer) -> None:
    """A critical response must still be able to score 1.0 with grounding down."""
    result = scorer.score(
        signals(
            grounding=(SignalStatus.UNAVAILABLE, 0.0),
            pii=(SignalStatus.FAIL, 1.0),
            safety=(SignalStatus.FAIL, 1.0),
            cost=(SignalStatus.FAIL, 1.0),
        ),
        WEIGHTS,
    )
    assert result.overall_score == pytest.approx(1.0)
    assert result.unavailable_checks == ["grounding"]
    assert any("renormalised" in note for note in result.notes)


def test_without_renormalisation_the_score_would_be_capped(scorer: RiskScorer) -> None:
    """Guards the specific bug renormalisation exists to prevent."""
    result = scorer.score(
        signals(
            grounding=(SignalStatus.UNAVAILABLE, 0.0),
            pii=(SignalStatus.FAIL, 1.0),
            safety=(SignalStatus.FAIL, 1.0),
            cost=(SignalStatus.FAIL, 1.0),
        ),
        WEIGHTS,
    )
    naive = 0.20 + 0.30 + 0.10  # what a fixed denominator would have produced
    assert result.overall_score > naive


def test_skipped_detector_is_excluded_not_treated_as_clean(scorer: RiskScorer) -> None:
    result = scorer.score(
        signals(
            grounding=(SignalStatus.SKIPPED, 0.0),
            pii=(SignalStatus.FAIL, 1.0),
        ),
        WEIGHTS,
    )
    # 0.20*1.0 / (0.30 + 0.20 + 0.10)
    assert result.overall_score == pytest.approx(0.3333, abs=1e-4)
    assert result.skipped_checks == ["grounding"]


def test_component_scores_are_recorded_even_when_excluded(scorer: RiskScorer) -> None:
    """The audit needs every component, including the ones that did not count."""
    result = scorer.score(
        signals(grounding=(SignalStatus.UNAVAILABLE, 0.0), safety=(SignalStatus.FAIL, 0.8)),
        WEIGHTS,
    )
    assert set(result.component_scores) == {"grounding", "pii", "safety", "bias", "cost"}
    assert result.weights == WEIGHTS


def test_nothing_usable_says_so_loudly(scorer: RiskScorer) -> None:
    """0.0 with every detector down must not be readable as 'safe'."""
    result = scorer.score(
        signals(
            grounding=(SignalStatus.UNAVAILABLE, 0.0),
            pii=(SignalStatus.UNAVAILABLE, 0.0),
            safety=(SignalStatus.UNAVAILABLE, 0.0),
            cost=(SignalStatus.UNAVAILABLE, 0.0),
        ),
        WEIGHTS,
    )
    assert result.overall_score == 0.0
    assert len(result.unavailable_checks) == 4
    assert any("not meaningful" in note for note in result.notes)


def test_missing_weights_fall_back_to_defaults(scorer: RiskScorer) -> None:
    result = scorer.score(signals(safety=(SignalStatus.FAIL, 1.0)), {"safety": 0.30})
    assert result.weights["grounding"] == DEFAULT_WEIGHTS["grounding"]


def test_score_is_clamped_to_unit_interval(scorer: RiskScorer) -> None:
    """Weights need not sum to 1.0; the output still must."""
    result = scorer.score(
        signals(
            grounding=(SignalStatus.FAIL, 1.0),
            pii=(SignalStatus.FAIL, 1.0),
            safety=(SignalStatus.FAIL, 1.0),
            cost=(SignalStatus.FAIL, 1.0),
        ),
        {"grounding": 5.0, "safety": 5.0, "pii": 5.0, "cost": 5.0},
    )
    assert 0.0 <= result.overall_score <= 1.0


def test_severity_is_not_folded_into_the_score(scorer: RiskScorer) -> None:
    """Severity travels on the signal for policy rules; it must not double-count in the math."""
    low = signals(safety=(SignalStatus.FAIL, 0.5))
    low.safety.severity = Severity.LOW
    high = signals(safety=(SignalStatus.FAIL, 0.5))
    high.safety.severity = Severity.CRITICAL
    assert scorer.score(low, WEIGHTS).overall_score == scorer.score(high, WEIGHTS).overall_score
