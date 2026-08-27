"""Deterministic demo fixtures loaded from ``data/demo/scenarios.yaml``.

Shared by three consumers so they can never drift apart:

* :class:`app.services.llm.mock_provider.MockProvider` — canned model responses;
* ``backend/tests/test_scenarios.py`` — expected decisions;
* ``scripts/run_demo.py`` — the judge-facing walkthrough.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

#: Sentinel in an expectation field meaning "do not assert on this".
ANY = "any"


def normalise(text: str) -> str:
    """Lowercase and collapse whitespace, so YAML folded blocks and real prompts match."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


class ScenarioResponse(BaseModel):
    """A canned model response, including simulated usage and latency.

    ``latency_ms`` is *reported*, not slept. Scenario E declares 9.6 s so the cost signal
    sees a realistic anomaly while the test suite still runs in milliseconds.
    """

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

    @property
    def clean_text(self) -> str:
        return re.sub(r"\s+", " ", self.text).strip()

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ScenarioExpectation(BaseModel):
    """What the pipeline should conclude. ``ANY`` means unasserted."""

    decision: str | None = None
    grounding_status: str | None = None
    pii_detected: bool | None = None
    pii_kinds: list[str] = Field(default_factory=list)
    safety_status: str | None = None
    safety_severity: str | None = None
    over_token_budget: bool | None = None
    over_latency_budget: bool | None = None
    requires_human_review: bool | None = None
    notes: str = ""


class Scenario(BaseModel):
    id: str
    title: str
    description: str = ""
    prompt: str
    match_keywords: list[str] = Field(default_factory=list)
    response: ScenarioResponse
    expect: ScenarioExpectation = Field(default_factory=ScenarioExpectation)

    def matches(self, prompt: str) -> bool:
        """Exact normalised prompt match, or every keyword present."""
        p = normalise(prompt)
        if p == normalise(self.prompt):
            return True
        if not self.match_keywords:
            return False
        return all(normalise(k) in p for k in self.match_keywords)


class ScenarioLibrary(BaseModel):
    version: int = 1
    domain: str = "unknown"
    system_prompt: str = ""
    fallback: ScenarioResponse
    scenarios: list[Scenario] = Field(default_factory=list)

    def match(self, prompt: str) -> Scenario | None:
        """First scenario whose prompt or keyword set matches. Exact matches win."""
        p = normalise(prompt)
        for sc in self.scenarios:
            if p == normalise(sc.prompt):
                return sc
        for sc in self.scenarios:
            if sc.matches(prompt):
                return sc
        return None

    def by_id(self, scenario_id: str) -> Scenario:
        for sc in self.scenarios:
            if sc.id == scenario_id:
                return sc
        raise KeyError(f"unknown scenario id: {scenario_id!r}")


def load_scenario_library(path: str | Path) -> ScenarioLibrary:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"scenario fixtures not found at {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return ScenarioLibrary.model_validate(raw)


@lru_cache(maxsize=4)
def cached_scenario_library(path: str) -> ScenarioLibrary:
    return load_scenario_library(path)


__all__ = [
    "ANY",
    "Scenario",
    "ScenarioExpectation",
    "ScenarioLibrary",
    "ScenarioResponse",
    "cached_scenario_library",
    "load_scenario_library",
    "normalise",
]
