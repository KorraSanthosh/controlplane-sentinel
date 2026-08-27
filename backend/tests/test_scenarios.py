"""End-to-end scenario test suite.

Asserts what the six demo cases (A through F) in ``data/demo/scenarios.yaml`` conclude
when run through the complete ControlPlane Sentinel pipeline.

Each scenario tests a distinct governance risk pattern:
- A: Grounded factual answer -> ALLOW
- B: Contradicted hallucination -> BLOCK
- C: Response leaking customer PII -> REDACT (masked delivery)
- D: Unauthorised financial commitment & confidential disclosure -> BLOCK (critical safety)
- E: Expensive, slow response -> ALLOW (with cost anomaly recorded)
- F: Unverified entity claim -> FLAG (requires human review)
"""

from __future__ import annotations

import pytest

from app.demo.scenarios import ANY, Scenario, ScenarioLibrary
from app.schemas.decision import Decision
from app.schemas.signals import GroundingStatus, Severity, SignalStatus
from tests.conftest import chat_request


async def test_scenario_a_grounded_safe(container, scenarios: ScenarioLibrary) -> None:
    sc = scenarios.by_id("A_grounded_safe")
    req = chat_request(sc.prompt)
    response, record = await container.orchestrator.process(req)

    assert response.decision is Decision.ALLOW
    assert response.risk.signals.grounding.grounding_status is GroundingStatus.GROUNDED
    assert not response.risk.signals.pii.detected
    assert response.risk.signals.safety.status is SignalStatus.PASS
    assert not response.requires_human_review
    assert not response.answer_modified
    assert record.decision is Decision.ALLOW


async def test_scenario_b_hallucination_contradicted(container, scenarios: ScenarioLibrary) -> None:
    sc = scenarios.by_id("B_hallucination_contradicted")
    req = chat_request(sc.prompt)
    response, record = await container.orchestrator.process(req)

    assert response.decision is Decision.BLOCK
    assert response.risk.signals.grounding.grounding_status is GroundingStatus.CONTRADICTED
    assert not response.risk.signals.pii.detected
    assert response.original_withheld
    assert response.answer_modified
    assert record.decision is Decision.BLOCK


async def test_scenario_c_pii_leakage(container, scenarios: ScenarioLibrary) -> None:
    sc = scenarios.by_id("C_pii_leakage")
    req = chat_request(sc.prompt)
    response, record = await container.orchestrator.process(req)

    assert response.decision is Decision.REDACT
    assert response.risk.signals.pii.detected
    expected_kinds = {"email", "phone", "account_number", "credit_card", "date_of_birth", "postal_address"}
    detected_kinds = set(response.risk.signals.pii.counts.keys())
    assert expected_kinds <= detected_kinds
    assert not response.original_withheld, "the customer still gets their answer masked"
    assert response.answer_modified
    assert record.decision is Decision.REDACT


async def test_scenario_d_unsafe_policy_violation(container, scenarios: ScenarioLibrary) -> None:
    sc = scenarios.by_id("D_unsafe_policy_violation")
    req = chat_request(sc.prompt)
    response, record = await container.orchestrator.process(req)

    assert response.decision is Decision.BLOCK
    assert response.risk.signals.safety.status is SignalStatus.FAIL
    assert response.risk.signals.safety.severity is Severity.CRITICAL
    assert response.original_withheld
    assert record.decision is Decision.BLOCK


async def test_scenario_e_cost_anomaly(container, scenarios: ScenarioLibrary) -> None:
    sc = scenarios.by_id("E_cost_anomaly")
    req = chat_request(sc.prompt)
    response, record = await container.orchestrator.process(req)

    assert response.decision is Decision.ALLOW
    assert response.risk.signals.cost.status is SignalStatus.WARN
    assert response.risk.signals.cost.over_token_budget
    assert response.risk.signals.cost.over_latency_budget
    assert "cost_anomaly" in [r.rule_id for r in response.fired_rules]
    assert record.decision is Decision.ALLOW


async def test_scenario_f_insufficient_evidence(container, scenarios: ScenarioLibrary) -> None:
    sc = scenarios.by_id("F_insufficient_evidence")
    req = chat_request(sc.prompt)
    response, record = await container.orchestrator.process(req)

    assert response.decision is Decision.FLAG
    assert response.risk.signals.grounding.grounding_status is GroundingStatus.UNSUPPORTED
    assert response.requires_human_review
    assert record.decision is Decision.FLAG


@pytest.mark.parametrize("scenario_id", ["A_grounded_safe", "B_hallucination_contradicted", "C_pii_leakage", "D_unsafe_policy_violation", "E_cost_anomaly", "F_insufficient_evidence"])
async def test_scenario_table_driven_validation(container, scenarios: ScenarioLibrary, scenario_id: str) -> None:
    sc = scenarios.by_id(scenario_id)
    req = chat_request(sc.prompt)
    response, record = await container.orchestrator.process(req)

    expect = sc.expect
    if expect.decision:
        assert response.decision == expect.decision

    if expect.grounding_status and expect.grounding_status != ANY:
        assert response.risk.signals.grounding.grounding_status.value == expect.grounding_status

    if expect.pii_detected is not None:
        assert response.risk.signals.pii.detected is expect.pii_detected

    if expect.pii_kinds:
        detected = set(response.risk.signals.pii.counts.keys())
        assert set(expect.pii_kinds) <= detected

    if expect.safety_status:
        assert response.risk.signals.safety.status.value == expect.safety_status

    if expect.safety_severity:
        assert response.risk.signals.safety.severity.value == expect.safety_severity

    if expect.over_token_budget is not None:
        assert response.risk.signals.cost.over_token_budget is expect.over_token_budget

    if expect.over_latency_budget is not None:
        assert response.risk.signals.cost.over_latency_budget is expect.over_latency_budget

    if expect.requires_human_review is not None:
        assert response.requires_human_review is expect.requires_human_review

    assert record.request_id == response.request_id
    assert record.decision == response.decision
