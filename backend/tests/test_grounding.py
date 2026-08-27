"""Grounding verification against the trusted graph.

The four statuses carry different meanings and must not collapse into each other:

* ``grounded`` — a stored fact agrees.
* ``contradicted`` — a stored fact disagrees. We can refute.
* ``unsupported`` — the graph holds nothing on the subject. We can neither confirm nor refute.
* ``unavailable`` — the graph could not be asked at all.

The last two are the ones a careless implementation merges into "looks fine", so they get the
most attention here. There is also a regression test per claim-extraction bug that produced a
wrong status, because each of those turned a correct answer into a spurious FLAG.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.schemas.signals import GroundingStatus, Severity, SignalStatus
from app.services.grounding.claims import extract_claims, split_sentences
from app.services.grounding.graph_repo import (
    GraphFact,
    GraphHealth,
    GraphRepository,
    GraphSeed,
    GraphUnavailable,
)
from app.services.grounding.memory_repo import InMemoryGraphRepository
from app.services.grounding.service import (
    SCORE_CONTRADICTED,
    SCORE_GROUNDED,
    SCORE_UNSUPPORTED,
    GroundingService,
)


@pytest.fixture
def repo(graph_seed: GraphSeed) -> InMemoryGraphRepository:
    return InMemoryGraphRepository(graph_seed)


@pytest.fixture
def grounding(repo: InMemoryGraphRepository) -> GroundingService:
    return GroundingService(repo)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class _DeadRepo(GraphRepository):
    """Every call fails, the way an unreachable Aura instance does."""

    backend = "neo4j"
    detail = "Neo4j query failed: ServiceUnavailable: cannot resolve address"

    async def all_entities(self):
        raise GraphUnavailable(self.detail)

    async def facts_for_entities(self, entity_ids):
        raise GraphUnavailable(self.detail)

    async def document_titles(self):
        raise GraphUnavailable(self.detail)

    async def health_check(self) -> GraphHealth:
        return GraphHealth(backend=self.backend, healthy=False, detail=self.detail)


class _FactsDieRepo(InMemoryGraphRepository):
    """Aliases load, then the connection drops before the facts query.

    This is the harder failure: claims have already been extracted, so the service is past
    the point where it could quietly decide there was nothing to check.
    """

    backend = "neo4j"

    async def facts_for_entities(self, entity_ids):
        raise GraphUnavailable("Neo4j query failed: SessionExpired")


class _NeoShapedRepo(GraphRepository):
    """A repository shaped the way the Neo4j one behaves.

    Three incidentals differ from the in-memory implementation: rows arrive in arbitrary
    order, numeric properties come back stringified by the driver (``str(r["object"])`` in
    :class:`Neo4jGraphRepository`, so an integer property can read as ``"40.0"``), and the
    result set may include facts the caller did not single out. If a verdict changes when only
    these change, grounding depends on the backend — which is what the two-implementation
    design promises it does not.
    """

    backend = "neo4j"

    def __init__(self, seed: GraphSeed) -> None:
        self._entities = list(reversed(seed.entities))
        self._facts = [
            f.model_copy(update={"object": f"{float(f.object)}"})
            if f.value_type == "number"
            else f
            for f in reversed(seed.facts)
        ]
        self._titles = {d.id: d.title for d in seed.documents}

    async def all_entities(self):
        return list(self._entities)

    async def facts_for_entities(self, entity_ids) -> list[GraphFact]:
        return list(self._facts)  # deliberately a superset, and unordered

    async def document_titles(self) -> dict[str, str]:
        return dict(self._titles)

    async def health_check(self) -> GraphHealth:
        return GraphHealth(backend=self.backend, healthy=True, detail="test double")


# ---------------------------------------------------------------------------
# GROUNDED
# ---------------------------------------------------------------------------
async def test_matching_fact_is_grounded(grounding: GroundingService) -> None:
    signal = await grounding.verify("The Plus plan includes 40 GB of high-speed data.")
    assert signal.status is SignalStatus.PASS
    assert signal.grounding_status is GroundingStatus.GROUNDED
    assert signal.score == SCORE_GROUNDED
    assert signal.severity is Severity.NONE
    assert (signal.claims_checked, signal.claims_grounded) == (1, 1)


async def test_grounded_claim_carries_citable_evidence(grounding: GroundingService) -> None:
    """A decision a human cannot check is not auditable, so evidence must name its source."""
    signal = await grounding.verify("The Plus plan includes 40 GB of high-speed data.")
    evidence = signal.claims[0].evidence[0]
    assert evidence.supports
    assert evidence.reference == "plan_plus-INCLUDED_DATA_GB-40"
    assert evidence.source_document == "Northwind Consumer Plan Schedule, January 2026"


async def test_the_nearest_subject_gets_the_value(grounding: GroundingService) -> None:
    """Three subjects, two values, one sentence.

    A first-value-wins extractor gives Premium the 14-day window and reports a contradiction
    that does not exist. Proximity assignment is what keeps this honest.
    """
    signal = await grounding.verify(
        "Basic and Plus carry a 14-day refund window, while Premium carries a "
        "30-day refund window."
    )
    assert signal.grounding_status is GroundingStatus.GROUNDED
    assert signal.claims_grounded == 3
    assert signal.claims_contradicted == 0


async def test_a_negated_restriction_is_grounded_not_contradicted(
    grounding: GroundingService,
) -> None:
    """The graph says Basic excludes roaming; so does the response. That is agreement."""
    signal = await grounding.verify("The Basic plan does not include international roaming.")
    assert signal.grounding_status is GroundingStatus.GROUNDED
    assert signal.claims[0].object == "false"


# ---------------------------------------------------------------------------
# CONTRADICTED
# ---------------------------------------------------------------------------
async def test_wrong_number_is_contradicted(grounding: GroundingService) -> None:
    signal = await grounding.verify("The Premium plan costs $95 per month.")
    assert signal.status is SignalStatus.FAIL
    assert signal.grounding_status is GroundingStatus.CONTRADICTED
    assert signal.score == SCORE_CONTRADICTED
    assert signal.severity is Severity.CRITICAL
    assert signal.claims_contradicted == 1


async def test_contradiction_shows_the_refuting_fact(grounding: GroundingService) -> None:
    signal = await grounding.verify("The Premium plan costs $95 per month.")
    evidence = signal.claims[0].evidence[0]
    assert evidence.supports is False
    assert evidence.object == "70"
    assert "70" in signal.claims[0].explanation


async def test_falsely_bundled_feature_is_contradicted(grounding: GroundingService) -> None:
    """Scenario B's core claim, and the reason a boolean fact is in the seed at all.

    The trailing "at no extra cost" must not read as a negation — treating it as one would
    turn the hallucination into an apparent match.
    """
    signal = await grounding.verify(
        "The Premium plan includes unlimited international roaming at no extra cost."
    )
    assert signal.grounding_status is GroundingStatus.CONTRADICTED
    assert signal.claims[0].object == "true"
    assert signal.claims[0].evidence[0].object == "false"


async def test_one_contradiction_outweighs_several_matches(grounding: GroundingService) -> None:
    """A response is not averaged into acceptability by the true things it also said."""
    signal = await grounding.verify(
        "The Premium plan includes 100 GB of data. The Premium plan costs $95 per month."
    )
    assert signal.claims_grounded == 1
    assert signal.claims_contradicted == 1
    assert signal.grounding_status is GroundingStatus.CONTRADICTED


# ---------------------------------------------------------------------------
# UNSUPPORTED
# ---------------------------------------------------------------------------
async def test_unknown_subject_is_unsupported_not_contradicted(
    grounding: GroundingService,
) -> None:
    """Absence of evidence is not evidence of falsehood — the graph is closed-world."""
    signal = await grounding.verify("The Enterprise Fiber tier includes 500 GB of data.")
    assert signal.status is SignalStatus.WARN
    assert signal.grounding_status is GroundingStatus.UNSUPPORTED
    assert signal.score == SCORE_UNSUPPORTED
    assert signal.severity is Severity.MEDIUM
    assert signal.claims_unsupported == 1
    assert signal.claims_contradicted == 0


async def test_unsupported_claim_names_the_subject_without_a_determiner(
    grounding: GroundingService,
) -> None:
    signal = await grounding.verify("The Enterprise Fiber tier includes 500 GB of data.")
    claim = signal.claims[0]
    assert claim.subject == "Enterprise Fiber"
    assert claim.evidence == []
    assert "neither confirmed nor refuted" in claim.explanation


async def test_contradiction_wins_over_unsupported_in_the_same_response(
    grounding: GroundingService,
) -> None:
    signal = await grounding.verify(
        "The Enterprise Fiber tier is our newest option. The Premium plan costs $95 per month."
    )
    assert signal.claims_unsupported == 1
    assert signal.claims_contradicted == 1
    assert signal.grounding_status is GroundingStatus.CONTRADICTED


# ---------------------------------------------------------------------------
# Regression: leading determiners are not part of a product name
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "The Plus plan includes 40 GB of high-speed data.",
        "The Basic plan costs $25 per month.",
        "Our Premium plan costs $70 per month.",
        "That Basic plan includes 10 GB.",
        "The Northwind Plus plan includes 40 GB.",
    ],
)
async def test_a_known_plan_is_never_reported_as_an_unknown_subject(
    grounding: GroundingService, text: str
) -> None:
    """A determiner swallowed into the captured phrase made "The Plus" look unrecognised.

    That produced an unsupported claim, an UNSUPPORTED status and a FLAG on answers that were
    entirely correct — the single most damaging extraction bug so far, so every determiner the
    demo prose actually uses is pinned here.
    """
    signal = await grounding.verify(text)
    assert signal.claims_unsupported == 0, f"spurious unknown subject in: {text}"
    assert signal.grounding_status is GroundingStatus.GROUNDED


def test_brand_prefix_is_optional_in_prose(graph_seed: GraphSeed) -> None:
    """"the Northwind Plus plan" and "the Plus plan" are the same product."""
    with_brand = extract_claims("The Northwind Plus plan includes 40 GB.", graph_seed.entities)
    without = extract_claims("The Plus plan includes 40 GB.", graph_seed.entities)
    assert [c.key() for c in with_brand.claims] == [c.key() for c in without.claims]


# ---------------------------------------------------------------------------
# SKIPPED — nothing verifiable was asserted
# ---------------------------------------------------------------------------
async def test_no_factual_claim_is_skipped_not_passed(grounding: GroundingService) -> None:
    """SKIPPED keeps grounding out of the weighted score instead of donating a clean 0."""
    signal = await grounding.verify(
        "I'm sorry about the trouble. Let me pull up your account and take a look."
    )
    assert signal.status is SignalStatus.SKIPPED
    assert signal.claims_checked == 0
    assert signal.score == 0.0
    assert not signal.usable


async def test_assertion_outside_graph_coverage_is_counted_not_judged(
    grounding: GroundingService,
) -> None:
    """The graph knows the roaming add-on's price, not its data allowance.

    Reporting that as contradicted would be a closed-world assumption dressed up as a finding.
    """
    signal = await grounding.verify("The Roaming Pack includes 5 GB of data.")
    assert signal.status is SignalStatus.SKIPPED
    assert signal.claims_checked == 0
    assert signal.claims_unverifiable == 1
    assert "outside the graph's coverage" in signal.explanation


async def test_empty_response_is_skipped(grounding: GroundingService) -> None:
    assert (await grounding.verify("")).status is SignalStatus.SKIPPED


# ---------------------------------------------------------------------------
# UNAVAILABLE (FR-11)
# ---------------------------------------------------------------------------
async def test_unreachable_graph_is_unavailable_never_pass() -> None:
    """The fail-safe this whole status exists for."""
    signal = await GroundingService(_DeadRepo()).verify("The Premium plan costs $95 per month.")
    assert signal.status is SignalStatus.UNAVAILABLE
    assert signal.grounding_status is GroundingStatus.UNAVAILABLE
    assert signal.status is not SignalStatus.PASS
    assert signal.severity is Severity.MEDIUM
    assert "ServiceUnavailable" in (signal.error or "")
    assert signal.graph_backend == "unavailable"
    assert not signal.usable


async def test_failure_after_extraction_is_still_unavailable(graph_seed: GraphSeed) -> None:
    """Claims were extracted, so the service cannot fall back on "nothing to check"."""
    signal = await GroundingService(_FactsDieRepo(graph_seed)).verify(
        "The Premium plan costs $95 per month."
    )
    assert signal.status is SignalStatus.UNAVAILABLE
    assert signal.claims_checked == 0
    assert "SessionExpired" in (signal.error or "")


async def test_unavailable_explanation_says_it_is_not_a_pass() -> None:
    """The wording is load-bearing: an auditor reads this line, not the enum."""
    signal = await GroundingService(_DeadRepo()).verify("The Premium plan costs $95.")
    assert "not as a pass" in signal.explanation


async def test_unavailable_scores_zero_and_is_excluded_from_aggregation() -> None:
    """0.0 here is not "no risk" — the scorer must drop the signal, not average it in."""
    signal = await GroundingService(_DeadRepo()).verify("The Premium plan costs $95.")
    assert signal.score == 0.0
    assert not signal.usable


def test_neo4j_repository_refuses_to_start_without_credentials(settings: Settings) -> None:
    """Selection happens in the container; this is the guard behind it."""
    from app.services.grounding.neo4j_repo import Neo4jGraphRepository

    assert not settings.neo4j_configured
    with pytest.raises(GraphUnavailable, match="CP_NEO4J_URI"):
        Neo4jGraphRepository(settings)


# ---------------------------------------------------------------------------
# Prescan (the fast-path probe)
# ---------------------------------------------------------------------------
async def test_prescan_extracts_without_querying_facts(graph_seed: GraphSeed) -> None:
    """Triage must be able to ask "is anything asserted here?" without paying for a lookup."""
    extraction, error = await GroundingService(_FactsDieRepo(graph_seed)).prescan(
        "The Premium plan costs $95 per month."
    )
    assert error is None
    assert extraction is not None and len(extraction.claims) == 1


async def test_prescan_reports_an_unreachable_graph() -> None:
    extraction, error = await GroundingService(_DeadRepo()).prescan("The Plus plan is $45.")
    assert extraction is None
    assert error and "ServiceUnavailable" in error


async def test_prescan_finds_nothing_in_a_non_factual_reply(grounding: GroundingService) -> None:
    extraction, _ = await grounding.prescan("Happy to help — what would you like to change?")
    assert extraction is not None and extraction.claims == []


async def test_verify_reuses_a_prescan_result(grounding: GroundingService) -> None:
    """The deep path must not re-extract; the same object drives both halves."""
    extraction, _ = await grounding.prescan("The Premium plan costs $95 per month.")
    signal = await grounding.verify("ignored because the extraction is supplied", extraction)
    assert signal.grounding_status is GroundingStatus.CONTRADICTED


# ---------------------------------------------------------------------------
# Backend parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The Plus plan includes 40 GB of high-speed data.", GroundingStatus.GROUNDED),
        ("The Premium plan costs $95 per month.", GroundingStatus.CONTRADICTED),
        ("The Enterprise Fiber tier includes 500 GB.", GroundingStatus.UNSUPPORTED),
    ],
)
async def test_verdict_is_the_same_whichever_backend_answers(
    graph_seed: GraphSeed, text: str, expected: GroundingStatus
) -> None:
    """Row order, driver-stringified numbers and a superset result set change nothing."""
    memory = await GroundingService(InMemoryGraphRepository(graph_seed)).verify(text)
    neo_shaped = await GroundingService(_NeoShapedRepo(graph_seed)).verify(text)
    assert memory.grounding_status is expected
    assert neo_shaped.grounding_status is expected
    assert neo_shaped.claims_checked == memory.claims_checked


async def test_the_answering_backend_is_recorded(
    grounding: GroundingService, graph_seed: GraphSeed
) -> None:
    """An audit that does not say where its evidence came from cannot be re-checked."""
    text = "The Plus plan includes 40 GB."
    assert (await grounding.verify(text)).graph_backend == "memory"
    assert (await GroundingService(_NeoShapedRepo(graph_seed)).verify(text)).graph_backend == (
        "neo4j"
    )


# ---------------------------------------------------------------------------
# Extraction internals
# ---------------------------------------------------------------------------
def test_sentences_are_split_on_terminators() -> None:
    assert split_sentences("One. Two! Three?  Four") == ["One.", "Two!", "Three?", "Four"]


def test_mentioned_entities_are_reported_for_the_facts_query(graph_seed: GraphSeed) -> None:
    result = extract_claims(
        "The Plus plan includes 40 GB. The Premium plan includes 100 GB.", graph_seed.entities
    )
    assert set(result.mentioned_entity_ids) == {"plan_plus", "plan_premium"}
    assert result.sentences_scanned == 2


def test_the_longest_alias_claims_the_span(graph_seed: GraphSeed) -> None:
    """"the premium plan" must not also register as a bare "premium" mention."""
    result = extract_claims("The Premium plan includes 100 GB.", graph_seed.entities)
    assert result.mentioned_entity_ids == ["plan_premium"]


def test_identical_claims_are_deduplicated(graph_seed: GraphSeed) -> None:
    result = extract_claims(
        "The Plus plan includes 40 GB. The Plus plan includes 40 GB.", graph_seed.entities
    )
    assert len(result.claims) == 1
