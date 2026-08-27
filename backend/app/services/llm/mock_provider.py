"""Deterministic mock LLM provider.

Backs the entire test suite and the offline demo. It performs no network I/O, so results
are byte-identical on every run — which is what makes the six scenarios assertable
(NFR-06) and keeps tests independent of a live model (NFR-05).
"""

from __future__ import annotations

from app.demo.scenarios import ScenarioLibrary
from app.services.llm.base import (
    LLMProvider,
    LLMResult,
    LLMUsage,
    ProviderHealth,
)

#: Markers the simulated judge treats as unsafe. Deliberately small and obvious — the real
#: judgement lives in the rule-based safety service; this only exercises the deep path
#: offline so the code path is demonstrably covered without a live model.
_SIMULATED_UNSAFE_MARKERS = (
    "internal margin",
    "no approval needed",
    "waive all outstanding charges",
    "rate will never increase",
)


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(
        self,
        library: ScenarioLibrary,
        model: str = "mock-controlplane-1",
    ) -> None:
        self.library = library
        self.model = model

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        scenario = self.library.match(prompt)
        canned = scenario.response if scenario else self.library.fallback
        return LLMResult(
            text=canned.clean_text,
            provider=self.name,
            model=self.model,
            usage=LLMUsage(
                input_tokens=canned.input_tokens,
                output_tokens=canned.output_tokens,
                total_tokens=canned.total_tokens,
            ),
            # Reported, not slept: keeps scenario E's 9.6 s cost anomaly realistic while
            # the suite still finishes in milliseconds.
            latency_ms=canned.latency_ms,
            stop_reason="end_turn",
            raw_metadata={
                "simulated": True,
                "scenario_id": scenario.id if scenario else None,
                "matched_fixture": scenario is not None,
            },
        )

    async def evaluate(
        self,
        instruction: str,
        content: str,
        *,
        max_tokens: int | None = None,
    ) -> LLMResult:
        lowered = content.lower()
        hits = [m for m in _SIMULATED_UNSAFE_MARKERS if m in lowered]
        verdict = "unsafe" if hits else "safe"
        return LLMResult(
            text=verdict,
            provider=self.name,
            model=self.model,
            usage=LLMUsage(input_tokens=len(content) // 4, output_tokens=1, total_tokens=len(content) // 4 + 1),
            latency_ms=15.0,
            stop_reason="end_turn",
            raw_metadata={"simulated": True, "markers": hits},
        )

    async def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            name=self.name,
            healthy=True,
            model=self.model,
            detail=(
                f"deterministic fixtures, {len(self.library.scenarios)} scenarios "
                f"({self.library.domain})"
            ),
        )

    def parse_usage(self, raw: object) -> LLMUsage:
        if isinstance(raw, dict):
            return LLMUsage(
                input_tokens=raw.get("input_tokens"),
                output_tokens=raw.get("output_tokens"),
                total_tokens=raw.get("total_tokens"),
            )
        return LLMUsage()


__all__ = ["MockProvider"]
