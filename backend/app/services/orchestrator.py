"""The ControlPlane pipeline.

One coherent path, not a bag of detectors (CLAUDE.md §4):

    request
      → model
      → FAST PATH   pii · safety rules · cost · claim pre-scan      always; no fact I/O
      → TRIAGE      is deeper analysis worth paying for?
      → DEEP PATH   graph verification · LLM judge                  only when triage says so
      → RISK        weighted aggregation over usable signals
      → DECISION    policy rules
      → ACTION      deliver · mask · flag · withhold
      → AUDIT       record returned to the caller for off-path persistence

Four properties this module is responsible for.

**Latency is protected by triage, and measured.** The fast path is regex and arithmetic. The two
expensive operations — a graph query and a second model call — are each gated on a cheap signal
saying they might matter. ``controlplane_overhead_ms`` is really measured on every request,
separately from the model's own latency, so the cost of governance is a number rather than a
claim.

**Escalation is evidence-driven, not blanket.** The judge can be reached two ways: a fast signal
already looks bad, or grounding came back contradicted/unsupported. In the first case the judge
runs concurrently with the graph query; in the second it runs after, because the thing that
justified it had not happened yet. A profile that wants neither simply sets no triggers.

**A detector that fails is not a detector that passed.** Every check runs inside a guard that
converts any exception into an ``UNAVAILABLE`` signal carrying the error. Unavailable signals are
excluded from the weighted score and handled explicitly by the policy fail-safe map, so a
crashed checker escalates rather than disappears (FR-11).

**The pipeline decides; it does not persist.** ``process`` returns the audit record alongside the
response and never writes it. The caller does that off the response path, which keeps a storage
outage from becoming a request failure.

The one failure this module does *not* absorb is the primary generation call. If the provider is
down there is no response, therefore no decision and no audit record; ``LLMUnavailable``
propagates and the API answers 503. Inventing a response to have something to govern would be
worse than reporting that the model is unreachable.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field

from app.core.config import Settings
from app.schemas.audit import AuditRecord
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.decision import Decision, DecisionResult
from app.schemas.signals import (
    CostSignal,
    GroundingSignal,
    GroundingStatus,
    PIISignal,
    RiskAssessment,
    RiskSignals,
    SafetySignal,
    Severity,
    SignalStatus,
    StageTiming,
    Telemetry,
)
from app.services.audit.service import AuditService
from app.services.cost.service import CostService
from app.services.grounding.claims import ExtractionResult
from app.services.grounding.service import GroundingService
from app.services.llm.base import LLMProvider, LLMResult
from app.services.pii.service import PIIService
from app.services.policy.decision import DecisionEngine
from app.services.policy.loader import PolicyProfile, PolicyRegistry
from app.services.risk.scoring import RiskScorer
from app.services.safety.service import SafetyService
from app.telemetry.timing import StageTimer

logger = logging.getLogger(__name__)

#: Timer entries that are the model's time, not ours, and are therefore excluded from the
#: control-plane overhead figure.
STAGE_LLM_REPORTED = "llm_reported_latency"
STAGE_LLM_WALLCLOCK = "llm_call_wallclock"
_MODEL_STAGES = (STAGE_LLM_REPORTED, STAGE_LLM_WALLCLOCK)


def _failure_detail(exc: BaseException, check: str) -> str:
    return f"{check} check failed: {type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class FastResult:
    """Everything the fast path produced, including the un-verified claim extraction."""

    pii: PIISignal
    safety: SafetySignal
    cost: CostSignal
    extraction: ExtractionResult | None
    prescan_error: str | None


@dataclass
class DeepPlan:
    """What triage decided to buy, and why. ``reasons`` is surfaced in the telemetry."""

    verify_grounding: bool = False
    run_judge: bool = False
    grounding_enabled: bool = False
    judge_available: bool = False
    reasons: list[str] = field(default_factory=list)

    #: Set when the judge was reached only because grounding came back bad. Recorded because
    #: that route costs an extra sequential round trip, and a latency metric that hid it would
    #: understate the worst case.
    judge_escalated_after_grounding: bool = False

    @property
    def took_deep_path(self) -> bool:
        return self.verify_grounding or self.run_judge


class Orchestrator:
    """Runs one request through the whole control plane."""

    def __init__(
        self,
        *,
        settings: Settings,
        llm: LLMProvider,
        grounding: GroundingService,
        pii: PIIService,
        safety: SafetyService,
        cost: CostService,
        scorer: RiskScorer,
        decision_engine: DecisionEngine,
        audit: AuditService,
        policies: PolicyRegistry,
        system_prompt: str | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.grounding = grounding
        self.pii = pii
        self.safety = safety
        self.cost = cost
        self.scorer = scorer
        self.decision_engine = decision_engine
        self.audit = audit
        self.policies = policies
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------ main
    async def process(
        self, request: ChatRequest, *, request_id: str | None = None
    ) -> tuple[ChatResponse, AuditRecord]:
        rid = request_id or uuid.uuid4().hex
        profile = self.policies.get(request.policy_profile)
        timer = StageTimer()

        # -- 1. generation -------------------------------------------------
        with timer.stage(STAGE_LLM_WALLCLOCK):
            result = await self.llm.generate(
                request.message,
                system=self.system_prompt,
                max_tokens=self.settings.llm_max_tokens,
            )
        original = result.text or ""
        # Recorded from the provider's own figure rather than our wall-clock: MockProvider
        # reports a scripted latency so the cost scenario is realistic without slowing tests.
        timer.record(STAGE_LLM_REPORTED, result.latency_ms)

        # -- 2. fast path --------------------------------------------------
        fast = await self._fast_path(original, request, profile, result, timer)
        pii_signal, safety_signal, cost_signal = fast.pii, fast.safety, fast.cost

        # -- 3. triage -----------------------------------------------------
        with timer.stage("triage"):
            fast_assessment = self._fast_assessment(
                profile, pii_signal, safety_signal, cost_signal
            )
            plan = self._triage(profile, fast_assessment, fast, safety_signal)

        # -- 4. deep path --------------------------------------------------
        grounding_signal, safety_signal = await self._deep_path(
            original, profile, plan, fast, safety_signal, timer
        )

        # -- 5. aggregate --------------------------------------------------
        signals = RiskSignals(
            grounding=grounding_signal,
            pii=pii_signal,
            safety=safety_signal,
            cost=cost_signal,
        )
        with timer.stage("risk_scoring"):
            assessment = self.scorer.score(signals, profile.weights)

        # -- 6. decide -----------------------------------------------------
        with timer.stage("decision"):
            decision = self.decision_engine.decide(assessment, profile)

        # -- 7. act --------------------------------------------------------
        with timer.stage("action"):
            delivered, withheld = self._apply_action(original, decision, profile, pii_signal)

        telemetry = self._telemetry(timer, result, plan)

        record = self.audit.build_record(
            request_id=rid,
            request=request,
            prompt=request.message,
            original_response=original,
            delivered_response=delivered,
            provider=result.provider,
            model=result.model,
            assessment=assessment,
            decision=decision,
            telemetry=telemetry,
        )

        reveal_original = bool(request.debug and self.settings.allow_debug_original)
        response = ChatResponse(
            request_id=rid,
            answer=delivered,
            decision=decision.decision,
            reason=decision.reason,
            original_answer=original if reveal_original else None,
            answer_modified=delivered != original,
            # True whenever the model's own text did not reach ``answer``, regardless of
            # whether a debug caller was also handed a copy of it.
            original_withheld=withheld,
            requires_human_review=decision.requires_human_review,
            risk=assessment,
            telemetry=telemetry,
            fired_rules=decision.fired_rules,
            provider=result.provider,
            model=result.model,
            policy_profile=profile.id,
        )
        return response, record

    # ------------------------------------------------------------- fast path
    async def _fast_path(
        self,
        text: str,
        request: ChatRequest,
        profile: PolicyProfile,
        result: LLMResult,
        timer: StageTimer,
    ) -> FastResult:
        """The always-on checks.

        Gathered rather than sequential. Three of the four are pure CPU work measured in
        microseconds, so concurrency buys little there — the one that matters is the claim
        pre-scan, which reads the entity alias table and so touches the graph on a cold cache.
        Structuring the stage this way means adding an I/O-bound fast check later needs no
        restructuring.
        """
        budget = profile.budget_for(request.use_case)

        with timer.stage("fast_path"):
            pii_signal, safety_signal, cost_signal, prescan = await asyncio.gather(
                self._run_pii(text, profile),
                self._run_safety_rules(text, profile),
                self._run_cost(result, budget.max_total_tokens, budget.max_latency_ms, profile),
                self._run_prescan(text, profile),
            )

        extraction, prescan_error = prescan
        return FastResult(
            pii=pii_signal,
            safety=safety_signal,
            cost=cost_signal,
            extraction=extraction,
            prescan_error=prescan_error,
        )

    async def _run_pii(self, text: str, profile: PolicyProfile) -> PIISignal:
        if not profile.is_enabled("pii"):
            return PIISignal(
                status=SignalStatus.SKIPPED,
                explanation="PII detection is disabled in this policy profile.",
            )
        try:
            return self.pii.scan(text)
        except Exception as exc:  # noqa: BLE001 - guard: unavailable, never pass
            logger.exception("PII scan failed")
            return PIISignal(
                status=SignalStatus.UNAVAILABLE,
                severity=Severity.MEDIUM,
                explanation=(
                    "PII detection could not run, so the response has not been cleared of "
                    "personal data. Reported as unavailable, not as a pass."
                ),
                error=_failure_detail(exc, "pii"),
            )

    async def _run_safety_rules(self, text: str, profile: PolicyProfile) -> SafetySignal:
        if not profile.is_enabled("safety"):
            return SafetySignal(
                status=SignalStatus.SKIPPED,
                explanation="Safety checking is disabled in this policy profile.",
            )
        categories = set(profile.safety_categories) if profile.safety_categories else None
        try:
            return self.safety.check_rules(text, enabled_categories=categories)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Safety rule evaluation failed")
            return SafetySignal(
                status=SignalStatus.UNAVAILABLE,
                severity=Severity.MEDIUM,
                explanation=(
                    "Safety rules could not be evaluated, so no policy violation has been "
                    "ruled out. Reported as unavailable, not as a pass."
                ),
                error=_failure_detail(exc, "safety"),
            )

    async def _run_cost(
        self,
        result: LLMResult,
        token_budget: int,
        latency_budget_ms: float,
        profile: PolicyProfile,
    ) -> CostSignal:
        if not profile.is_enabled("cost"):
            return CostSignal(
                status=SignalStatus.SKIPPED,
                explanation="Cost telemetry is disabled in this policy profile.",
                llm_latency_ms=result.latency_ms,
            )
        try:
            return self.cost.evaluate(
                result, token_budget=token_budget, latency_budget_ms=latency_budget_ms
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Cost evaluation failed")
            return CostSignal(
                status=SignalStatus.UNAVAILABLE,
                explanation="Cost and latency could not be evaluated.",
                error=_failure_detail(exc, "cost"),
                llm_latency_ms=result.latency_ms,
            )

    async def _run_prescan(
        self, text: str, profile: PolicyProfile
    ) -> tuple[ExtractionResult | None, str | None]:
        """Claim extraction only — no fact lookup. Returns ``(extraction, error)``."""
        if not profile.is_enabled("grounding") or not profile.triage.deep_grounding_enabled:
            return None, None
        try:
            return await self.grounding.prescan(text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Claim pre-scan failed")
            return None, _failure_detail(exc, "grounding pre-scan")

    # ---------------------------------------------------------------- triage
    def _fast_assessment(
        self,
        profile: PolicyProfile,
        pii_signal: PIISignal,
        safety_signal: SafetySignal,
        cost_signal: CostSignal,
    ) -> RiskAssessment:
        """Risk from the fast signals alone, with grounding marked SKIPPED.

        Reuses :class:`RiskScorer` rather than summing weights inline, so the renormalisation
        rule (a signal that did not score is out of the denominator) is defined in exactly one
        place. Duplicating it here would let triage act on a systematically different number
        from the one the policy engine later sees.
        """
        return self.scorer.score(
            RiskSignals(
                grounding=self._grounding_placeholder("Deferred to triage."),
                pii=pii_signal,
                safety=safety_signal,
                cost=cost_signal,
            ),
            profile.weights,
        )

    def _triage(
        self,
        profile: PolicyProfile,
        fast: RiskAssessment,
        fast_result: FastResult,
        safety_signal: SafetySignal,
    ) -> DeepPlan:
        triage = profile.triage
        plan = DeepPlan(
            grounding_enabled=profile.is_enabled("grounding") and triage.deep_grounding_enabled,
            judge_available=(
                profile.is_enabled("safety")
                and triage.judge_enabled
                and self.safety.llm is not None
            ),
        )

        extraction = fast_result.extraction
        if not plan.grounding_enabled or extraction is None:
            # Either the profile switched grounding off, or the graph is already known to be
            # unreachable — in which case the signal becomes UNAVAILABLE and the fail-safe
            # handles it. Neither case buys a fact query.
            pass
        elif extraction.claims:
            plan.verify_grounding = True
            plan.reasons.append(
                f"{len(extraction.claims)} verifiable claim(s) extracted from "
                f"{extraction.sentences_scanned} sentence(s); querying the trusted graph for "
                f"supporting or contradicting evidence"
            )
        elif not triage.deep_grounding_on_claims:
            plan.verify_grounding = True
            plan.reasons.append("this profile verifies grounding unconditionally")

        if plan.judge_available:
            if safety_signal.status in triage.judge_on_safety_status:
                plan.run_judge = True
                plan.reasons.append(
                    f"safety rule layer returned '{safety_signal.status.value}'; escalating to "
                    f"the LLM judge for a second opinion"
                )
            elif fast.overall_score >= triage.judge_on_fast_risk_at_or_above:
                plan.run_judge = True
                plan.reasons.append(
                    f"fast-path risk {fast.overall_score:.2f} reached this profile's judge "
                    f"threshold of {triage.judge_on_fast_risk_at_or_above:.2f}"
                )

        return plan

    # ------------------------------------------------------------- deep path
    async def _deep_path(
        self,
        text: str,
        profile: PolicyProfile,
        plan: DeepPlan,
        fast_result: FastResult,
        safety_signal: SafetySignal,
        timer: StageTimer,
    ) -> tuple[GroundingSignal, SafetySignal]:
        if not plan.took_deep_path:
            return self._grounding_without_query(plan, fast_result), safety_signal

        with timer.stage("deep_path"):
            # Genuinely concurrent when both are wanted: one is a database round trip, the
            # other a model call.
            jobs: list[str] = []
            coros = []
            if plan.verify_grounding:
                jobs.append("grounding")
                coros.append(self._run_deep_grounding(text, fast_result.extraction))
            if plan.run_judge:
                jobs.append("judge")
                coros.append(self._run_judge(text, safety_signal))

            outcome = dict(zip(jobs, await asyncio.gather(*coros)))

            grounding_signal = outcome.get("grounding") or self._grounding_without_query(
                plan, fast_result
            )
            safety_signal = outcome.get("judge", safety_signal)

            # Second escalation route: grounding has now resolved, and this profile treats the
            # result it produced as grounds for a judge opinion the fast signals did not
            # justify. Sequential by necessity — the trigger did not exist until now.
            if (
                not plan.run_judge
                and plan.judge_available
                and grounding_signal.grounding_status in profile.triage.judge_on_grounding_status
            ):
                plan.run_judge = True
                plan.judge_escalated_after_grounding = True
                plan.reasons.append(
                    f"grounding returned '{grounding_signal.grounding_status.value}'; this "
                    f"profile escalates that to the LLM judge"
                )
                safety_signal = await self._run_judge(text, safety_signal)

        return grounding_signal, safety_signal

    def _grounding_without_query(
        self, plan: DeepPlan, fast_result: FastResult
    ) -> GroundingSignal:
        """The grounding signal for a request that never reached the fact store."""
        if not plan.grounding_enabled:
            return self._grounding_placeholder(
                "Grounding verification is disabled in this policy profile."
            )

        if fast_result.extraction is None:
            return GroundingSignal(
                status=SignalStatus.UNAVAILABLE,
                grounding_status=GroundingStatus.UNAVAILABLE,
                severity=Severity.MEDIUM,
                explanation=(
                    "Grounding could not be verified because the trusted graph was "
                    "unreachable. Reported as unavailable, not as a pass."
                ),
                error=fast_result.prescan_error or "graph unavailable",
                graph_backend="unavailable",
            )

        # Nothing verifiable was asserted, so no fact query was needed. This is the fast path
        # staying fast, not a check that failed — hence SKIPPED, which keeps a misleadingly
        # clean 0.0 out of the weighted score.
        return GroundingSignal(
            status=SignalStatus.SKIPPED,
            grounding_status=GroundingStatus.GROUNDED,
            claims_checked=0,
            explanation=(
                f"No verifiable factual claim was found in "
                f"{fast_result.extraction.sentences_scanned} sentence(s), so the trusted graph "
                f"was not queried."
            ),
            graph_backend=self.grounding.repo.backend,
        )

    async def _run_deep_grounding(
        self, text: str, extraction: ExtractionResult | None
    ) -> GroundingSignal:
        try:
            return await self.grounding.verify(text, extraction)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Grounding verification failed")
            return GroundingSignal(
                status=SignalStatus.UNAVAILABLE,
                grounding_status=GroundingStatus.UNAVAILABLE,
                severity=Severity.MEDIUM,
                explanation=(
                    "Grounding verification failed, so the claims in this response are "
                    "unverified. Reported as unavailable, not as a pass."
                ),
                error=_failure_detail(exc, "grounding"),
                graph_backend="unavailable",
            )

    async def _run_judge(self, text: str, base: SafetySignal) -> SafetySignal:
        try:
            return await self.safety.judge(text, base)
        except Exception as exc:  # noqa: BLE001 - SafetyService guards too; belt and braces
            logger.exception("Safety judge failed")
            return base.model_copy(
                update={"judge_used": False, "error": _failure_detail(exc, "safety judge")}
            )

    # ---------------------------------------------------------------- action
    def _apply_action(
        self,
        original: str,
        decision: DecisionResult,
        profile: PolicyProfile,
        pii_signal: PIISignal,
    ) -> tuple[str, bool]:
        """Turn a decision into the text actually delivered. Returns ``(text, withheld)``."""
        if decision.decision is Decision.BLOCK:
            return profile.blocked_response, True

        # Not keyed on ``Decision.REDACT``: a FLAG that also carried PII must still be masked.
        if decision.apply_redaction and pii_signal.matches:
            return self.pii.redact(original, pii_signal.matches), False

        return original, False

    # ------------------------------------------------------------- telemetry
    def _telemetry(self, timer: StageTimer, result: LLMResult, plan: DeepPlan) -> Telemetry:
        overhead = timer.sum_except(*_MODEL_STAGES)
        return Telemetry(
            llm_latency_ms=round(result.latency_ms, 2),
            controlplane_overhead_ms=round(overhead, 2),
            total_ms=round(result.latency_ms + overhead, 2),
            path_taken="deep" if plan.took_deep_path else "fast",
            deep_path_reasons=list(plan.reasons),
            judge_escalated_after_grounding=plan.judge_escalated_after_grounding,
            stages=[
                StageTiming(stage=s.stage, duration_ms=round(s.duration_ms, 3))
                for s in timer.stages
            ],
            llm_latency_simulated=bool(result.raw_metadata.get("simulated", False)),
        )

    # ----------------------------------------------------------------- utils
    def _grounding_placeholder(self, why: str) -> GroundingSignal:
        """A SKIPPED grounding signal, so it is excluded from aggregation."""
        return GroundingSignal(
            status=SignalStatus.SKIPPED,
            grounding_status=GroundingStatus.GROUNDED,
            explanation=why,
            graph_backend=self.grounding.repo.backend,
        )


__all__ = [
    "STAGE_LLM_REPORTED",
    "STAGE_LLM_WALLCLOCK",
    "DeepPlan",
    "FastResult",
    "Orchestrator",
]
