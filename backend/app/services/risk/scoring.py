"""Risk aggregation (FR-08).

Combines component signals into one overall score using configurable weights.

The one subtle decision here: **weights are renormalised over the signals that actually
produced a usable score.** If grounding (weight 0.40) is unavailable, leaving its weight in
the denominator would cap the achievable score at 0.60 — a response with critical PII and a
critical safety violation would score lower than it should, precisely when the system knows
least. Renormalising keeps the score meaningful, and the *fact* of unavailability is handled
separately and explicitly by the policy fail-safe rules rather than by quietly moving a
number.
"""

from __future__ import annotations

from app.schemas.signals import RiskAssessment, RiskSignals

#: Fallback weights, used only if a policy profile omits them. Prototype configuration.
DEFAULT_WEIGHTS: dict[str, float] = {
    "grounding": 0.40,
    "safety": 0.30,
    "pii": 0.20,
    "bias": 0.00,
    "cost": 0.10,
}



class RiskScorer:
    name = "risk"

    def score(
        self,
        signals: RiskSignals,
        weights: dict[str, float] | None = None,
    ) -> RiskAssessment:
        weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        by_name = signals.as_dict()

        component_scores: dict[str, float] = {}
        unavailable: list[str] = []
        skipped: list[str] = []
        notes: list[str] = []

        usable_weight = 0.0
        weighted_total = 0.0

        for name, signal in by_name.items():
            component_scores[name] = signal.score
            weight = float(weights.get(name, 0.0))

            if signal.status.value == "unavailable":
                unavailable.append(name)
                continue
            if signal.status.value == "skipped":
                skipped.append(name)
                continue

            usable_weight += weight
            weighted_total += weight * signal.score

        if usable_weight > 0:
            overall = weighted_total / usable_weight
        else:
            # Nothing could be scored. Report 0.0 but say so loudly — the policy layer must
            # not read this as "safe".
            overall = 0.0
            notes.append(
                "No detector produced a usable score; the overall score is not meaningful "
                "and the fail-safe policy applies."
            )

        if unavailable:
            notes.append(
                f"Weights renormalised over available detectors; unavailable: "
                f"{', '.join(unavailable)}."
            )
        if skipped:
            notes.append(f"Not applicable to this response: {', '.join(skipped)}.")

        return RiskAssessment(
            overall_score=round(min(1.0, max(0.0, overall)), 4),
            component_scores={k: round(v, 4) for k, v in component_scores.items()},
            weights=weights,
            unavailable_checks=unavailable,
            skipped_checks=skipped,
            signals=signals,
            notes=notes,
        )


__all__ = ["DEFAULT_WEIGHTS", "RiskScorer"]
