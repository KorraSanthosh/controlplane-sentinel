"""Composition root.

Everything is constructed exactly once, here. Three consumers share it — the FastAPI app, the
test suite and ``scripts/run_demo.py`` — which is the point: the demo and the tests exercise the
same wiring the server does, not a parallel arrangement that happens to behave similarly.

Backend selection (Mongo vs memory, Neo4j vs memory, real model vs mock) is delegated to the
per-service factories and always resolves to *something runnable*. A checkout with no
credentials boots and serves all six scenarios; ``/health`` reports which backend actually
answered, so a fallback is visible rather than silent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.demo.scenarios import ScenarioLibrary, cached_scenario_library
from app.services.audit.factory import build_audit_repository
from app.services.audit.repository import AuditRepository
from app.services.audit.service import AuditService
from app.services.cost.service import CostService
from app.services.grounding.factory import build_graph_repository
from app.services.grounding.graph_repo import GraphRepository
from app.services.grounding.service import GroundingService
from app.services.llm.base import LLMProvider
from app.services.llm.factory import build_llm_provider
from app.services.orchestrator import Orchestrator
from app.services.pii.service import PIIService
from app.services.policy.decision import DecisionEngine
from app.services.policy.loader import PolicyRegistry, load_policy_registry
from app.services.risk.scoring import RiskScorer
from app.services.safety.service import SafetyService, cached_safety_rules

logger = logging.getLogger(__name__)

#: The safety rule corpus lives alongside the policy profiles but is not one.
SAFETY_RULES_FILE = "safety_rules.yaml"


@dataclass
class Container:
    settings: Settings
    scenarios: ScenarioLibrary
    policies: PolicyRegistry
    llm: LLMProvider
    graph: GraphRepository
    audit_repo: AuditRepository
    pii: PIIService
    safety: SafetyService
    cost: CostService
    grounding: GroundingService
    scorer: RiskScorer
    decision_engine: DecisionEngine
    audit: AuditService
    orchestrator: Orchestrator

    async def aclose(self) -> None:
        """Release every held connection. Failures are logged, never raised — shutdown must
        not be the thing that takes the process down."""
        for name, closable in (
            ("llm", self.llm),
            ("graph", self.graph),
            ("audit", self.audit_repo),
        ):
            try:
                await closable.aclose()
            except Exception:  # noqa: BLE001
                logger.warning("error closing %s backend", name, exc_info=True)


async def build_container(settings: Settings | None = None) -> Container:
    settings = settings or get_settings()

    # Policies are loaded first and eagerly. A profile that does not parse is a hard startup
    # failure: serving traffic under a policy we could not fully read is worse than not serving.
    policies = load_policy_registry(
        settings.policy_dir_abs, settings.default_policy_profile
    )
    scenarios = cached_scenario_library(str(settings.scenarios_path_abs))
    safety_rules = cached_safety_rules(str(settings.policy_dir_abs / SAFETY_RULES_FILE))

    llm = build_llm_provider(settings)
    graph = await build_graph_repository(settings)
    audit_repo = await build_audit_repository(settings)

    pii = PIIService()
    # The judge shares the primary provider. Deliberate for the prototype — a real deployment
    # would likely point the judge at a cheaper model, which is a constructor argument away.
    safety = SafetyService(safety_rules, llm=llm)
    cost = CostService(settings)
    grounding = GroundingService(graph)
    scorer = RiskScorer()
    decision_engine = DecisionEngine()
    audit = AuditService(audit_repo, pii)

    orchestrator = Orchestrator(
        settings=settings,
        llm=llm,
        grounding=grounding,
        pii=pii,
        safety=safety,
        cost=cost,
        scorer=scorer,
        decision_engine=decision_engine,
        audit=audit,
        policies=policies,
        system_prompt=scenarios.system_prompt or None,
    )

    logger.info(
        "ControlPlane ready — llm=%s graph=%s audit=%s policies=%s",
        llm.name,
        graph.backend,
        audit_repo.backend,
        ",".join(policies.ids()),
    )

    return Container(
        settings=settings,
        scenarios=scenarios,
        policies=policies,
        llm=llm,
        graph=graph,
        audit_repo=audit_repo,
        pii=pii,
        safety=safety,
        cost=cost,
        grounding=grounding,
        scorer=scorer,
        decision_engine=decision_engine,
        audit=audit,
        orchestrator=orchestrator,
    )


__all__ = ["SAFETY_RULES_FILE", "Container", "build_container"]
