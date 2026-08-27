"""The pipeline, end to end.

``test_scenarios.py`` asserts *what* the six demo cases decide. This file asserts *how* the
pipeline reaches a decision — the properties a scenario table cannot see:

* a detector that fails reports ``UNAVAILABLE`` and never ``PASS`` (FR-11);
* the two expensive operations are bought only when triage says so — the graph query is
  literally counted, so "we protect latency" is a measurement rather than a claim;
* both judge escalation routes work, and the sequential one is visible in the telemetry;
* control-plane overhead is measured separately from the model's own latency;
* a decision becomes delivered text correctly, including masking a response that was flagged
  rather than redacted.

The *response* is the input under test here, not the prompt, so most tests drive a scripted
provider that returns exactly the text they care about instead of routing through the demo
fixtures. Wiring otherwise comes from the real container, so a broken composition root fails
these tests too.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.container import Container
from app.schemas.decision import Decision
from app.schemas.signals import GroundingStatus, Severity, SignalStatus
from app.services.audit.service import sha256_text
from app.services.grounding.graph_repo import GraphSeed
from app.services.grounding.memory_repo import InMemoryGraphRepository
from app.services.grounding.service import GroundingService
from app.services.llm.base import LLMProvider, LLMResult, LLMUnavailable, LLMUsage, ProviderHealth
from app.services.llm.mock_provider import MockProvider
from app.services.orchestrator import (
    STAGE_LLM_REPORTED,
    STAGE_LLM_WALLCLOCK,
    Orchestrator,
)
from app.services.pii.service import REDACTION_TEMPLATE
from app.services.policy.loader import PolicyProfile, PolicyRegistry
from app.services.safety.service import SafetyService
from tests.conftest import chat_request

# ---------------------------------------------------------------------------
# Response texts, each chosen for the single path it exercises
# ---------------------------------------------------------------------------
#: One verifiable claim that the seed graph agrees with.
GROUNDED = "The Plus plan includes 40 GB of high-speed data per month."

#: Asserts nothing checkable, so it must never pay for a graph round trip.
NO_CLAIM = "I'm sorry about the trouble. Let me pull up your file and take a look."

#: The graph records $70 for Premium.
CONTRADICTED = "The Premium plan costs $95 per month."

#: A subject the graph has never heard of — unsupported, not false.
UNSUPPORTED = "The Enterprise Fiber tier includes 500 GB of data."

#: Trips POL-COMMITMENT-01 at critical severity, and the mock judge agrees it is unsafe.
UNSAFE = "I'll waive all outstanding charges right now, no approval needed."

#: One maskable identifier and nothing else.
PII_ONLY = "You can reach the account holder at j.doe@northwind.example.com."

#: Unsupported *and* PII-bearing: the decision is FLAG, but the email must still be masked.
UNSUPPORTED_WITH_PII = (
    "The Enterprise Fiber tier includes 500 GB of data. "
    "Write to j.doe@northwind.example.com for the contract."
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class _ScriptedProvider(LLMProvider):
    """Returns exactly the text the test asked for, with scripted usage and latency."""

    name = "scripted"

    def __init__(
        self,
        text: str,
        *,
        input_tokens: int = 150,
        output_tokens: int = 90,
        latency_ms: float = 420.0,
        simulated: bool = True,
    ) -> None:
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.simulated = simulated

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        return LLMResult(
            text=self.text,
            provider=self.name,
            model="scripted-1",
            usage=LLMUsage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                total_tokens=self.input_tokens + self.output_tokens,
            ),
            latency_ms=self.latency_ms,
            stop_reason="end_turn",
            raw_metadata={"simulated": self.simulated},
        )

    async def evaluate(self, instruction: str, content: str, *, max_tokens: int | None = None):
        # The judge is wired through SafetyService's own provider. If the primary provider is
        # ever asked to judge, that is a wiring bug and should be loud.
        raise AssertionError("the primary provider must not be used as the safety judge")

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(name=self.name, healthy=True, model="scripted-1")


class _DeadProvider(_ScriptedProvider):
    """Generation itself fails — the one failure the pipeline does not absorb."""

    name = "dead"

    async def generate(self, prompt: str, **kwargs) -> LLMResult:
        raise LLMUnavailable("Anthropic API unreachable: connection refused")


class _CountingJudge(MockProvider):
    """Mock provider that counts judge calls, so "the judge did not run" is assertable."""

    def __init__(self, library) -> None:
        super().__init__(library)
        self.evaluate_calls = 0

    async def evaluate(self, instruction: str, content: str, *, max_tokens: int | None = None):
        self.evaluate_calls += 1
        return await super().evaluate(instruction, content, max_tokens=max_tokens)


class _CountingGraph(InMemoryGraphRepository):
    """Counts the expensive half of grounding: the fact query.

    ``all_entities`` is the cheap alias table the fast-path pre-scan reads; only
    ``facts_for_entities`` is the round trip triage exists to avoid.
    """

    def __init__(self, seed: GraphSeed) -> None:
        super().__init__(seed)
        self.fact_queries = 0

    async def facts_for_entities(self, entity_ids):
        self.fact_queries += 1
        return await super().facts_for_entities(entity_ids)


def _explode(*_args, **_kwargs):
    """Raise at call time, so one synchronous stub can stand in for an async detector too."""
    raise RuntimeError("detector exploded")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pipeline_for(container: Container, **overrides) -> Orchestrator:
    """An orchestrator wired from the real container, with the named parts swapped out."""
    parts = {
        "settings": container.settings,
        "llm": container.llm,
        "grounding": container.grounding,
        "pii": container.pii,
        "safety": container.safety,
        "bias": container.bias,
        "cost": container.cost,
        "scorer": container.scorer,
        "decision_engine": container.decision_engine,
        "audit": container.audit,
        "policies": container.policies,
        "system_prompt": None,
    }
    return Orchestrator(**{**parts, **overrides})



def only(profile: PolicyProfile, **overrides) -> PolicyRegistry:
    """A single-profile registry with fields overridden — no YAML round trip needed."""
    tuned = replace(profile, **overrides)
    return PolicyRegistry(profiles={tuned.id: tuned}, default_id=tuned.id)


def judged_by(container: Container, judge: MockProvider) -> SafetyService:
    """The container's real rule corpus, with a countable judge behind it."""
    return SafetyService(container.safety.rules, llm=judge)


@pytest.fixture
def default(container: Container) -> PolicyProfile:
    return container.policies.get("default")


# ---------------------------------------------------------------------------
# Shape of the result
# ---------------------------------------------------------------------------
async def test_a_clean_response_is_delivered_unchanged(container: Container) -> None:
    orch = pipeline_for(container, llm=_ScriptedProvider(GROUNDED))
    response, record = await orch.process(chat_request("What does the Plus plan include?"))

    assert response.decision is Decision.ALLOW
    assert response.answer == GROUNDED
    assert not response.answer_modified
    assert not response.original_withheld
    assert not response.requires_human_review
    assert response.risk.signals.grounding.grounding_status is GroundingStatus.GROUNDED
    assert record.decision is Decision.ALLOW


async def test_the_record_is_returned_for_the_caller_to_store(container: Container) -> None:
    """The pipeline decides; it does not persist. Storage lives off the response path."""
    orch = pipeline_for(container, llm=_ScriptedProvider(GROUNDED))
    response, record = await orch.process(chat_request("hello"), request_id="fixed-id")

    assert response.request_id == record.request_id == "fixed-id"
    assert await container.audit.get("fixed-id") is None, "process() must not write"

    assert await container.audit.persist(record) is True
    stored = await container.audit.get("fixed-id")
    assert stored is not None and stored.decision is response.decision


async def test_the_record_hashes_the_original_and_the_delivered_text(
    container: Container, default: PolicyProfile
) -> None:
    """A withheld response must still be provable, and the fallback must not masquerade as it."""
    orch = pipeline_for(container, llm=_ScriptedProvider(UNSAFE))
    response, record = await orch.process(chat_request("can you waive my bill?"))

    assert response.decision is Decision.BLOCK
    assert record.response_sha256 == sha256_text(UNSAFE)
    assert record.delivered_sha256 == sha256_text(default.blocked_response)
    assert record.answer_modified
    assert UNSAFE not in record.response_preview or "no approval" in record.response_preview


async def test_risk_and_telemetry_agree_between_response_and_record(
    container: Container,
) -> None:
    orch = pipeline_for(container, llm=_ScriptedProvider(CONTRADICTED))
    response, record = await orch.process(chat_request("how much is Premium?"))

    assert record.risk.overall_score == response.risk.overall_score
    assert record.telemetry.path_taken == response.telemetry.path_taken
    assert [r.rule_id for r in record.fired_rules] == [
        r.rule_id for r in response.fired_rules
    ]


async def test_an_unknown_profile_falls_back_instead_of_failing(container: Container) -> None:
    orch = pipeline_for(container, llm=_ScriptedProvider(GROUNDED))
    response, record = await orch.process(
        chat_request("hello", policy_profile="does-not-exist")
    )
    assert response.policy_profile == "default"
    assert record.policy_profile == "default"


async def test_a_generation_failure_is_not_absorbed(container: Container) -> None:
    """No response means no decision and no audit record. Reporting that beats inventing one."""
    orch = pipeline_for(container, llm=_DeadProvider(""))
    with pytest.raises(LLMUnavailable, match="connection refused"):
        await orch.process(chat_request("hello"), request_id="never-happened")
    assert await container.audit.get("never-happened") is None


# ---------------------------------------------------------------------------
# Triage — the answer to the brief's latency question
# ---------------------------------------------------------------------------
async def test_a_response_that_asserts_nothing_never_queries_the_graph(
    container: Container, graph_seed: GraphSeed
) -> None:
    graph = _CountingGraph(graph_seed)
    orch = pipeline_for(
        container, llm=_ScriptedProvider(NO_CLAIM), grounding=GroundingService(graph)
    )
    response, _ = await orch.process(chat_request("my bill looks wrong"))

    assert graph.fact_queries == 0, "an unverifiable reply must not pay for graph I/O"
    assert response.telemetry.path_taken == "fast"
    assert response.telemetry.deep_path_reasons == []
    assert response.risk.signals.grounding.status is SignalStatus.SKIPPED
    assert "grounding" in response.risk.skipped_checks
    assert response.decision is Decision.ALLOW


async def test_a_verifiable_claim_does_buy_the_graph_query(
    container: Container, graph_seed: GraphSeed
) -> None:
    graph = _CountingGraph(graph_seed)
    orch = pipeline_for(
        container, llm=_ScriptedProvider(GROUNDED), grounding=GroundingService(graph)
    )
    response, _ = await orch.process(chat_request("what does Plus include?"))

    assert graph.fact_queries == 1
    assert response.telemetry.path_taken == "deep"
    assert any("verifiable claim" in r for r in response.telemetry.deep_path_reasons)


async def test_a_profile_can_switch_the_graph_query_off_entirely(
    container: Container, default: PolicyProfile, graph_seed: GraphSeed
) -> None:
    """The latency escape hatch, and its cost stated out loud.

    With grounding off, the contradiction that would otherwise BLOCK this response is never
    discovered and it is delivered. That is the trade the profile is making, and it is visible
    in ``skipped_checks`` rather than hidden behind an ALLOW.
    """
    graph = _CountingGraph(graph_seed)
    orch = pipeline_for(
        container,
        llm=_ScriptedProvider(CONTRADICTED),
        grounding=GroundingService(graph),
        policies=only(default, triage=replace(default.triage, deep_grounding_enabled=False)),
    )
    response, _ = await orch.process(chat_request("how much is Premium?"))

    assert graph.fact_queries == 0
    assert response.telemetry.path_taken == "fast"
    assert response.risk.signals.grounding.status is SignalStatus.SKIPPED
    assert "grounding" in response.risk.skipped_checks
    assert response.decision is Decision.ALLOW


async def test_unconditional_grounding_still_short_circuits_on_nothing_to_check(
    container: Container, default: PolicyProfile, graph_seed: GraphSeed
) -> None:
    """``deep_grounding_on_claims: false`` opts into the deep path, not into pointless I/O.

    Triage plans the verification, so the request is correctly reported as deep; the grounding
    service then finds no claim to check and returns before the fact query. Two independent
    thrift mechanisms, and the second one holds even when the first is switched off.
    """
    graph = _CountingGraph(graph_seed)
    orch = pipeline_for(
        container,
        llm=_ScriptedProvider(NO_CLAIM),
        grounding=GroundingService(graph),
        policies=only(default, triage=replace(default.triage, deep_grounding_on_claims=False)),
    )
    response, _ = await orch.process(chat_request("my bill looks wrong"))

    assert response.telemetry.path_taken == "deep"
    assert any("unconditionally" in r for r in response.telemetry.deep_path_reasons)
    assert graph.fact_queries == 0
    assert response.risk.signals.grounding.status is SignalStatus.SKIPPED


# ---------------------------------------------------------------------------
# The judge, and its two routes in
# ---------------------------------------------------------------------------
async def test_the_judge_is_not_bought_by_a_clean_response(container: Container) -> None:
    """A deep-path request is not automatically a judged request — the gates are separate."""
    judge = _CountingJudge(container.scenarios)
    orch = pipeline_for(
        container, llm=_ScriptedProvider(GROUNDED), safety=judged_by(container, judge)
    )
    response, _ = await orch.process(chat_request("what does Plus include?"))

    assert judge.evaluate_calls == 0
    assert response.telemetry.path_taken == "deep"  # the graph was asked; the judge was not
    assert not response.risk.signals.safety.judge_used


async def test_a_failing_rule_layer_escalates_to_the_judge(container: Container) -> None:
    judge = _CountingJudge(container.scenarios)
    orch = pipeline_for(
        container, llm=_ScriptedProvider(UNSAFE), safety=judged_by(container, judge)
    )
    response, _ = await orch.process(chat_request("can you waive my bill?"))
    safety = response.risk.signals.safety

    assert judge.evaluate_calls == 1
    assert safety.judge_used and safety.judge_verdict == "unsafe"
    assert "JUDGE-01" in {v.policy_id for v in safety.violations}
    assert safety.severity is Severity.CRITICAL
    assert not response.telemetry.judge_escalated_after_grounding, "this route is concurrent"
    assert response.decision is Decision.BLOCK


async def test_grounding_can_escalate_to_the_judge_after_the_fact(
    container: Container,
) -> None:
    """The sequential route: nothing cheap was worried, then the graph refuted a claim.

    It costs an extra round trip after the graph query, so telemetry names it separately —
    a latency figure that folded it in with the concurrent route would understate the worst
    case.
    """
    judge = _CountingJudge(container.scenarios)
    orch = pipeline_for(
        container, llm=_ScriptedProvider(CONTRADICTED), safety=judged_by(container, judge)
    )
    response, _ = await orch.process(chat_request("how much is Premium?"))
    safety = response.risk.signals.safety

    assert judge.evaluate_calls == 1
    assert response.telemetry.judge_escalated_after_grounding
    assert any(
        "escalates that to the LLM judge" in r for r in response.telemetry.deep_path_reasons
    )
    # The judge found no policy violation, and a clean judge verdict cannot manufacture one.
    assert safety.judge_used and safety.status is SignalStatus.PASS
    assert response.decision is Decision.BLOCK  # on the contradiction, not on the judge


async def test_a_profile_can_switch_the_judge_off(
    container: Container, default: PolicyProfile
) -> None:
    """With the judge disabled the rule layer is still authoritative, and still blocks."""
    judge = _CountingJudge(container.scenarios)
    orch = pipeline_for(
        container,
        llm=_ScriptedProvider(UNSAFE),
        safety=judged_by(container, judge),
        policies=only(default, triage=replace(default.triage, judge_enabled=False)),
    )
    response, _ = await orch.process(chat_request("can you waive my bill?"))

    assert judge.evaluate_calls == 0
    assert not response.risk.signals.safety.judge_used
    assert response.telemetry.path_taken == "fast"
    assert response.decision is Decision.BLOCK


async def test_a_judge_outage_does_not_downgrade_the_rule_verdict(
    container: Container,
) -> None:
    """The judge is a second opinion, never a veto on the deterministic layer."""

    class _DeadJudge(_CountingJudge):
        async def evaluate(self, instruction, content, *, max_tokens=None):
            self.evaluate_calls += 1
            raise RuntimeError("judge endpoint timed out")

    judge = _DeadJudge(container.scenarios)
    orch = pipeline_for(
        container, llm=_ScriptedProvider(UNSAFE), safety=judged_by(container, judge)
    )
    response, _ = await orch.process(chat_request("can you waive my bill?"))
    safety = response.risk.signals.safety

    assert judge.evaluate_calls == 1
    assert not safety.judge_used
    assert safety.status is SignalStatus.FAIL
    assert safety.severity is Severity.CRITICAL
    assert "timed out" in (safety.error or "")
    assert response.decision is Decision.BLOCK


# ---------------------------------------------------------------------------
# Fail-safe: a detector that failed is not a detector that passed (FR-11)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("check", "method", "text", "expected"),
    [
        ("pii", "scan", GROUNDED, Decision.FLAG),
        ("safety", "check_rules", GROUNDED, Decision.FLAG),
        ("cost", "evaluate", GROUNDED, Decision.ALLOW),
        ("grounding", "verify", GROUNDED, Decision.FLAG),
        ("grounding", "prescan", GROUNDED, Decision.FLAG),
    ],
)
async def test_a_failing_detector_is_unavailable_never_pass(
    container: Container,
    monkeypatch: pytest.MonkeyPatch,
    check: str,
    method: str,
    text: str,
    expected: Decision,
) -> None:
    monkeypatch.setattr(getattr(container, check), method, _explode)
    orch = pipeline_for(container, llm=_ScriptedProvider(text))
    response, record = await orch.process(chat_request("what does Plus include?"))

    signal = response.risk.signals.model_dump()[check]
    assert signal["status"] == SignalStatus.UNAVAILABLE.value
    assert signal["status"] != SignalStatus.PASS.value
    assert check in response.risk.unavailable_checks
    assert check in record.unavailable_checks
    # ``cost`` is the one detector this profile is willing to lose, and it says so out loud.
    assert response.decision is expected
    assert f"failsafe.{check}_unavailable" in [r.rule_id for r in response.fired_rules]


async def test_the_error_reaches_the_signal_not_just_the_log(container: Container, monkeypatch) -> None:
    monkeypatch.setattr(container.grounding, "verify", _explode)
    orch = pipeline_for(container, llm=_ScriptedProvider(GROUNDED))
    response, _ = await orch.process(chat_request("what does Plus include?"))
    grounding = response.risk.signals.grounding

    assert "RuntimeError" in (grounding.error or "")
    assert "detector exploded" in (grounding.error or "")
    assert grounding.grounding_status is GroundingStatus.UNAVAILABLE
    assert not grounding.usable
    assert "not as a pass" in grounding.explanation


async def test_an_unavailable_detector_does_not_cap_the_risk_score(
    container: Container, monkeypatch
) -> None:
    """Weights are renormalised, so losing grounding does not make an unsafe answer look safer."""
    monkeypatch.setattr(container.grounding, "prescan", _explode)
    monkeypatch.setattr(container.grounding, "verify", _explode)
    orch = pipeline_for(container, llm=_ScriptedProvider(UNSAFE))
    response, _ = await orch.process(chat_request("can you waive my bill?"))

    assert "grounding" in response.risk.unavailable_checks
    # safety scores 1.0 with weight 0.30 out of a usable 0.60 → 0.50, not 0.30.
    assert response.risk.overall_score == pytest.approx(0.5, abs=0.01)
    assert any("renormalised" in n for n in response.risk.notes)


async def test_a_broken_pii_scanner_does_not_leak_into_the_audit_preview(
    container: Container, monkeypatch
) -> None:
    """Masking is what makes a preview safe to store; if it breaks, the preview must go."""
    monkeypatch.setattr(container.pii, "scan", _explode)
    orch = pipeline_for(container, llm=_ScriptedProvider(PII_ONLY))
    _, record = await orch.process(chat_request("what is my email on file?"))

    assert "j.doe@northwind.example.com" not in record.response_preview
    assert "masking failed" in record.response_preview


async def test_a_disabled_check_is_skipped_not_unavailable(
    container: Container, default: PolicyProfile
) -> None:
    """SKIPPED and UNAVAILABLE must not collapse: one is a choice, the other is a failure."""
    off = only(default, enabled_checks={n: False for n in ("grounding", "pii", "safety", "bias", "cost")})
    orch = pipeline_for(container, llm=_ScriptedProvider(PII_ONLY), policies=off)
    response, _ = await orch.process(chat_request("what is my email on file?"))

    assert sorted(response.risk.skipped_checks) == ["bias", "cost", "grounding", "pii", "safety"]
    assert response.risk.unavailable_checks == []
    assert response.risk.overall_score == 0.0
    # A profile that checks nothing catches nothing. The point is that the emptiness is stated.
    assert any("not meaningful" in n for n in response.risk.notes)
    assert response.answer == PII_ONLY
    assert response.decision is Decision.ALLOW


# ---------------------------------------------------------------------------
# Turning a decision into delivered text
# ---------------------------------------------------------------------------
async def test_block_delivers_the_governed_fallback(
    container: Container, default: PolicyProfile
) -> None:
    orch = pipeline_for(container, llm=_ScriptedProvider(UNSAFE))
    response, _ = await orch.process(chat_request("can you waive my bill?"))

    assert response.decision is Decision.BLOCK
    assert response.answer == default.blocked_response
    assert response.original_withheld and response.answer_modified
    assert "waive" not in response.answer
    assert response.original_answer is None, "debug was not requested"


async def test_redact_masks_and_still_delivers(container: Container) -> None:
    orch = pipeline_for(container, llm=_ScriptedProvider(PII_ONLY))
    response, _ = await orch.process(chat_request("what is my email on file?"))

    assert response.decision is Decision.REDACT
    assert "j.doe@northwind.example.com" not in response.answer
    assert REDACTION_TEMPLATE.format(kind="EMAIL") in response.answer
    assert response.answer.startswith("You can reach the account holder at ")
    assert response.answer_modified
    assert not response.original_withheld, "the customer still got their answer"
    assert response.telemetry.path_taken == "fast"


async def test_a_flagged_response_is_still_masked(container: Container) -> None:
    """Masking is orthogonal to the tier. A FLAG is delivered, so its PII must be gone."""
    orch = pipeline_for(container, llm=_ScriptedProvider(UNSUPPORTED_WITH_PII))
    response, _ = await orch.process(chat_request("tell me about Enterprise Fiber"))

    assert response.decision is Decision.FLAG
    assert response.requires_human_review
    assert "j.doe@northwind.example.com" not in response.answer
    assert "Enterprise Fiber tier includes 500 GB" in response.answer
    fired = [r.rule_id for r in response.fired_rules]
    assert {"insufficient_evidence", "pii_redactable"} <= set(fired)


async def test_debug_reveals_the_original_only_when_the_server_allows_it(
    container: Container,
) -> None:
    orch = pipeline_for(container, llm=_ScriptedProvider(UNSAFE))
    allowed, _ = await orch.process(chat_request("waive it", debug=True))
    assert allowed.original_answer == UNSAFE
    assert allowed.original_withheld, "a debug copy is not delivery"

    locked = pipeline_for(
        container,
        llm=_ScriptedProvider(UNSAFE),
        settings=container.settings.model_copy(update={"allow_debug_original": False}),
    )
    response, _ = await locked.process(chat_request("waive it", debug=True))
    assert response.original_answer is None
    assert response.answer == container.policies.get("default").blocked_response


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------
async def test_overhead_is_measured_separately_from_model_latency(
    container: Container,
) -> None:
    """The number that answers "what does governance cost?" — ours, not the model's."""
    orch = pipeline_for(container, llm=_ScriptedProvider(GROUNDED, latency_ms=9600.0))
    response, _ = await orch.process(chat_request("what does Plus include?"))
    t = response.telemetry

    assert t.llm_latency_ms == 9600.0
    assert 0.0 < t.controlplane_overhead_ms < 1000.0, "in-process detectors; generous ceiling"
    assert t.total_ms == pytest.approx(t.llm_latency_ms + t.controlplane_overhead_ms, abs=0.05)

    ours = sum(
        s.duration_ms
        for s in t.stages
        if s.stage not in (STAGE_LLM_REPORTED, STAGE_LLM_WALLCLOCK)
    )
    assert ours == pytest.approx(t.controlplane_overhead_ms, abs=0.05)


async def test_every_stage_is_named(container: Container) -> None:
    orch = pipeline_for(container, llm=_ScriptedProvider(GROUNDED))
    response, _ = await orch.process(chat_request("what does Plus include?"))
    named = [s.stage for s in response.telemetry.stages]

    assert {
        STAGE_LLM_WALLCLOCK,
        STAGE_LLM_REPORTED,
        "fast_path",
        "triage",
        "deep_path",
        "risk_scoring",
        "decision",
        "action",
    } <= set(named)
    assert named.index("fast_path") < named.index("triage") < named.index("deep_path")


async def test_the_fast_path_records_no_deep_stage(container: Container) -> None:
    orch = pipeline_for(container, llm=_ScriptedProvider(NO_CLAIM))
    response, _ = await orch.process(chat_request("my bill looks wrong"))
    assert "deep_path" not in {s.stage for s in response.telemetry.stages}


async def test_simulated_model_latency_is_labelled_as_such(container: Container) -> None:
    """A 9.6-second demo number must not imply wall-clock was measured."""
    simulated = pipeline_for(container, llm=_ScriptedProvider(GROUNDED, simulated=True))
    real = pipeline_for(container, llm=_ScriptedProvider(GROUNDED, simulated=False))

    assert (await simulated.process(chat_request("x")))[0].telemetry.llm_latency_simulated
    assert not (await real.process(chat_request("x")))[0].telemetry.llm_latency_simulated


async def test_a_cost_overrun_is_recorded_without_changing_the_decision(
    container: Container,
) -> None:
    """Blocking a correct answer for verbosity is the alert fatigue this system avoids."""
    orch = pipeline_for(
        container,
        llm=_ScriptedProvider(GROUNDED, input_tokens=3400, output_tokens=1200, latency_ms=9600.0),
    )
    response, _ = await orch.process(chat_request("what does Plus include?"))
    cost = response.risk.signals.cost

    assert cost.status is SignalStatus.WARN
    assert cost.over_token_budget and cost.over_latency_budget
    assert cost.total_tokens == 4600
    assert "cost_anomaly" in [r.rule_id for r in response.fired_rules]
    assert response.decision is Decision.ALLOW
    assert response.answer == GROUNDED


# ---------------------------------------------------------------------------
# Bias Signal Integration
# ---------------------------------------------------------------------------
async def test_bias_signal_triggers_flag_decision(container: Container) -> None:
    biased_text = "Please note that female applicants require spouse approval for verification."
    orch = pipeline_for(container, llm=_ScriptedProvider(biased_text))
    response, record = await orch.process(chat_request("Can I apply for an account?"))

    assert response.decision is Decision.FLAG
    assert response.requires_human_review
    assert response.risk.signals.bias.status is SignalStatus.FAIL
    assert response.risk.signals.bias.severity is Severity.HIGH
    assert "gender" in response.risk.signals.bias.groups_implicated
    assert "high_bias_risk" in [r.rule_id for r in response.fired_rules]
    assert record.decision is Decision.FLAG


async def test_bias_check_failure_reports_unavailable(container: Container) -> None:
    class _FailingBiasService:
        def check(self, text, enabled_categories=None):
            raise RuntimeError("bias engine failure")

    orch = pipeline_for(container, llm=_ScriptedProvider(GROUNDED), bias=_FailingBiasService())
    response, record = await orch.process(chat_request("hello"))

    assert response.risk.signals.bias.status is SignalStatus.UNAVAILABLE
    assert response.risk.signals.bias.severity is Severity.MEDIUM
    assert "bias" in response.risk.unavailable_checks
    assert response.decision is Decision.FLAG

