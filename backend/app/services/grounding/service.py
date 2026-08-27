"""Grounding verification.

Compares claims extracted from a model response against trusted graph facts and produces a
:class:`GroundingSignal`.

The four statuses mean distinct things, and the distinction is the point of the module:

* ``grounded`` — every checked claim matches a stored fact.
* ``contradicted`` — the graph holds a fact with a *different value*. We can refute.
* ``unsupported`` — the graph holds nothing about the subject. We cannot confirm or refute,
  so we do not pretend to. Absence of evidence is not evidence of falsehood.
* ``unavailable`` — the graph could not be queried at all. Never silently downgraded to
  "no contradiction found" (FR-11).

This provides evidence of agreement or disagreement with a trusted source. It does not prove
truth, and it says nothing about causality.
"""

from __future__ import annotations

import logging
import time

from app.schemas.signals import (
    Claim,
    Evidence,
    GroundingSignal,
    GroundingStatus,
    Severity,
    SignalStatus,
)
from app.services.grounding.claims import ExtractedClaim, ExtractionResult, extract_claims
from app.services.grounding.graph_repo import (
    GraphFact,
    GraphRepository,
    GraphUnavailable,
)

logger = logging.getLogger(__name__)

#: Risk scores per grounding outcome. Prototype policy configuration, documented in the
#: README — deliberately NOT presented as an industry standard.
SCORE_CONTRADICTED = 0.95
SCORE_UNSUPPORTED = 0.55
SCORE_GROUNDED = 0.0

_TRUE_TOKENS = {"true", "yes", "included", "1"}
_FALSE_TOKENS = {"false", "no", "excluded", "0"}


def _values_match(claim_value: str, fact: GraphFact) -> bool:
    """Compare a claimed value with a stored fact, by declared value type."""
    cv = (claim_value or "").strip().lower()
    fv = (fact.object or "").strip().lower()

    if fact.value_type == "number":
        try:
            return abs(float(cv) - float(fv)) < 1e-9
        except ValueError:
            return cv == fv
    if fact.value_type == "boolean":
        cb = cv in _TRUE_TOKENS if cv in _TRUE_TOKENS | _FALSE_TOKENS else None
        fb = fv in _TRUE_TOKENS if fv in _TRUE_TOKENS | _FALSE_TOKENS else None
        if cb is None or fb is None:
            return cv == fv
        return cb == fb
    return cv == fv


def _evidence(fact: GraphFact, supports: bool, doc_titles: dict[str, str]) -> Evidence:
    return Evidence(
        subject=fact.subject,
        predicate=fact.predicate,
        object=fact.object,
        reference=fact.reference(),
        supports=supports,
        source_document=doc_titles.get(fact.source_document or "", fact.source_document),
    )


class GroundingService:
    name = "grounding"

    def __init__(self, repo: GraphRepository) -> None:
        self.repo = repo

    def _unavailable(self, detail: str, duration_ms: float) -> GroundingSignal:
        return GroundingSignal(
            status=SignalStatus.UNAVAILABLE,
            grounding_status=GroundingStatus.UNAVAILABLE,
            score=0.0,  # excluded from aggregation; the policy fail-safe decides the action
            severity=Severity.MEDIUM,
            explanation=(
                "Grounding could not be verified because the trusted graph was unreachable. "
                "This is reported as unavailable, not as a pass."
            ),
            error=detail,
            graph_backend="unavailable",
            duration_ms=duration_ms,
        )

    async def prescan(self, text: str) -> tuple[ExtractionResult | None, str | None]:
        """Cheap triage probe: extract claims without touching fact storage.

        Returns ``(extraction, error)``. This is the fast-path half of grounding — pure
        regex over a cached alias table, no fact query. The orchestrator uses it to decide
        whether the expensive graph lookup is worth doing at all: a response that asserts
        nothing verifiable never pays for a graph round trip.
        """
        try:
            entities = await self.repo.all_entities()
        except GraphUnavailable as exc:
            return None, str(exc)
        return extract_claims(text or "", entities), None

    async def verify(
        self, text: str, extraction: ExtractionResult | None = None
    ) -> GroundingSignal:
        started = time.perf_counter()

        if extraction is None:
            extraction, error = await self.prescan(text)
            if extraction is None:
                return self._unavailable(
                    error or "graph unavailable", (time.perf_counter() - started) * 1000.0
                )

        if not extraction.claims:
            # Nothing verifiable was said. Reported as SKIPPED so it is excluded from the
            # weighted risk score rather than contributing a misleading clean 0.
            return GroundingSignal(
                status=SignalStatus.SKIPPED,
                grounding_status=GroundingStatus.GROUNDED,
                score=SCORE_GROUNDED,
                severity=Severity.NONE,
                claims_checked=0,
                explanation=(
                    f"No verifiable factual claims were extracted from "
                    f"{extraction.sentences_scanned} sentence(s); grounding not applicable."
                ),
                graph_backend=self.repo.backend,
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        try:
            facts = await self.repo.facts_for_entities(extraction.mentioned_entity_ids)
            doc_titles = await self.repo.document_titles()
        except GraphUnavailable as exc:
            return self._unavailable(str(exc), (time.perf_counter() - started) * 1000.0)

        fact_index: dict[tuple[str, str], GraphFact] = {
            (f.subject, f.predicate): f for f in facts
        }

        checked: list[Claim] = []
        grounded = unsupported = contradicted = unverifiable = 0

        for ec in extraction.claims:
            if ec.subject_id is None:
                unsupported += 1
                checked.append(
                    Claim(
                        text=ec.sentence,
                        subject=ec.subject_name,
                        predicate=None,
                        object=None,
                        status=GroundingStatus.UNSUPPORTED,
                        confidence=0.6,
                        evidence=[],
                        explanation=(
                            f"The trusted graph holds no entity matching "
                            f"'{ec.subject_name}'. The claim can be neither confirmed nor "
                            f"refuted, so it is treated as unsupported rather than false."
                        ),
                    )
                )
                continue

            if ec.predicate is None:
                continue

            fact = fact_index.get((ec.subject_id, ec.predicate))
            if fact is None:
                # Outside the graph's coverage for this subject. Counted, not judged.
                unverifiable += 1
                continue

            if _values_match(ec.value or "", fact):
                grounded += 1
                checked.append(
                    Claim(
                        text=ec.sentence,
                        subject=ec.subject_name,
                        predicate=ec.predicate,
                        object=ec.value,
                        status=GroundingStatus.GROUNDED,
                        confidence=0.95,
                        evidence=[_evidence(fact, True, doc_titles)],
                        explanation=(
                            f"{ec.subject_name} {ec.label} of {ec.value} matches the trusted "
                            f"value {fact.object}."
                        ),
                    )
                )
            else:
                contradicted += 1
                checked.append(
                    Claim(
                        text=ec.sentence,
                        subject=ec.subject_name,
                        predicate=ec.predicate,
                        object=ec.value,
                        status=GroundingStatus.CONTRADICTED,
                        confidence=0.9,
                        evidence=[_evidence(fact, False, doc_titles)],
                        explanation=(
                            f"Response states {ec.subject_name} {ec.label} is '{ec.value}', "
                            f"but the trusted source records '{fact.object}'."
                        ),
                    )
                )

        duration_ms = (time.perf_counter() - started) * 1000.0
        claims_checked = grounded + unsupported + contradicted

        if claims_checked == 0:
            return GroundingSignal(
                status=SignalStatus.SKIPPED,
                grounding_status=GroundingStatus.GROUNDED,
                score=SCORE_GROUNDED,
                severity=Severity.NONE,
                claims_checked=0,
                claims_unverifiable=unverifiable,
                explanation=(
                    f"{unverifiable} assertion(s) fell outside the graph's coverage for their "
                    f"subject; nothing verifiable to check."
                ),
                graph_backend=self.repo.backend,
                duration_ms=duration_ms,
            )

        if contradicted:
            status, gstatus = SignalStatus.FAIL, GroundingStatus.CONTRADICTED
            score, severity = SCORE_CONTRADICTED, Severity.CRITICAL
            explanation = (
                f"{contradicted} of {claims_checked} checked claim(s) contradict trusted "
                f"evidence."
            )
        elif unsupported:
            status, gstatus = SignalStatus.WARN, GroundingStatus.UNSUPPORTED
            score, severity = SCORE_UNSUPPORTED, Severity.MEDIUM
            explanation = (
                f"{unsupported} of {claims_checked} checked claim(s) have no supporting "
                f"evidence in the trusted graph. Evidence is insufficient to confirm or "
                f"refute."
            )
        else:
            status, gstatus = SignalStatus.PASS, GroundingStatus.GROUNDED
            score, severity = SCORE_GROUNDED, Severity.NONE
            explanation = f"All {grounded} checked claim(s) are supported by trusted evidence."

        if unverifiable:
            explanation += f" {unverifiable} further assertion(s) fell outside graph coverage."

        return GroundingSignal(
            status=status,
            grounding_status=gstatus,
            score=score,
            severity=severity,
            claims=checked,
            claims_checked=claims_checked,
            claims_grounded=grounded,
            claims_unsupported=unsupported,
            claims_contradicted=contradicted,
            claims_unverifiable=unverifiable,
            explanation=explanation,
            graph_backend=self.repo.backend,
            duration_ms=duration_ms,
        )


__all__ = [
    "SCORE_CONTRADICTED",
    "SCORE_GROUNDED",
    "SCORE_UNSUPPORTED",
    "GroundingService",
]
