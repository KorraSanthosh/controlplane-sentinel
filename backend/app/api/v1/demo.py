"""``/demo`` — the six scenario fixtures, exposed so the dashboard can offer them as one click.

The prompts are the same ones the mock provider matches on and the same ones the test suite
asserts against, read from the same file. A judge clicking "Contradicted fact" in the UI is
running exactly the case ``test_scenarios.py`` proves.

``expect`` is included deliberately. Showing what the scenario is *supposed* to conclude next to
what it actually concluded is more useful than a demo that only ever shows agreement — and it
means a regression is visible in the UI rather than only in CI.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import ScenariosDep
from app.demo.scenarios import Scenario, ScenarioExpectation

router = APIRouter(tags=["demo"])


class ScenarioView(BaseModel):
    id: str
    title: str
    description: str
    prompt: str
    expect: ScenarioExpectation
    #: Reported by the fixture, not measured. Surfaced so nobody reads the 9.6 s cost scenario
    #: as a real network measurement.
    simulated_input_tokens: int
    simulated_output_tokens: int
    simulated_latency_ms: float

    @classmethod
    def of(cls, sc: Scenario) -> "ScenarioView":
        return cls(
            id=sc.id,
            title=sc.title,
            description=sc.description,
            prompt=sc.prompt,
            expect=sc.expect,
            simulated_input_tokens=sc.response.input_tokens,
            simulated_output_tokens=sc.response.output_tokens,
            simulated_latency_ms=sc.response.latency_ms,
        )


class ScenarioList(BaseModel):
    domain: str
    note: str = (
        "Northwind Telecom is fictional. Responses, token counts and latencies are simulated "
        "fixtures, which is what makes these six cases reproducible."
    )
    scenarios: list[ScenarioView] = Field(default_factory=list)


@router.get("/demo/scenarios", response_model=ScenarioList, summary="The demo scenario fixtures")
async def list_scenarios(scenarios: ScenariosDep) -> ScenarioList:
    return ScenarioList(
        domain=scenarios.domain,
        scenarios=[ScenarioView.of(sc) for sc in scenarios.scenarios],
    )


@router.get(
    "/demo/scenarios/{scenario_id}",
    response_model=ScenarioView,
    summary="One demo scenario",
)
async def get_scenario(scenario_id: str, scenarios: ScenariosDep) -> ScenarioView:
    try:
        return ScenarioView.of(scenarios.by_id(scenario_id))
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No scenario '{scenario_id}'. Available: "
                f"{', '.join(s.id for s in scenarios.scenarios)}."
            ),
        ) from None


__all__ = ["ScenarioList", "ScenarioView", "router"]
