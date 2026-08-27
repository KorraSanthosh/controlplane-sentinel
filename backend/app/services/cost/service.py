"""Cost and latency telemetry (FR-07).

Turns provider usage and wall-clock into a risk signal. Note the deliberate asymmetry with
the other detectors: cost risk is *operational*, not ethical. A verbose but accurate and safe
answer should raise a cost flag, not be blocked — blocking correct answers for being
expensive is precisely the alert-fatigue failure mode the Round 2 brief calls out.
"""

from __future__ import annotations

import time

from app.core.config import Settings
from app.schemas.signals import CostSignal, Severity, SignalStatus
from app.services.llm.base import LLMResult


class CostService:
    name = "cost"

    def __init__(self, settings: Settings) -> None:
        self.price_input_per_mtok = settings.price_input_per_mtok
        self.price_output_per_mtok = settings.price_output_per_mtok

    def estimate_cost_usd(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        """None when usage is incomplete — a fabricated 0.0 would understate spend."""
        if input_tokens is None or output_tokens is None:
            return None
        return round(
            (input_tokens / 1_000_000) * self.price_input_per_mtok
            + (output_tokens / 1_000_000) * self.price_output_per_mtok,
            6,
        )

    def evaluate(
        self,
        result: LLMResult,
        *,
        token_budget: int,
        latency_budget_ms: float,
    ) -> CostSignal:
        started = time.perf_counter()
        usage = result.usage

        total = usage.total_tokens
        if total is None and usage.input_tokens is not None and usage.output_tokens is not None:
            total = usage.input_tokens + usage.output_tokens

        cost = self.estimate_cost_usd(usage.input_tokens, usage.output_tokens)

        token_ratio = (total / token_budget) if (total is not None and token_budget > 0) else 0.0
        latency_ratio = (
            (result.latency_ms / latency_budget_ms) if latency_budget_ms > 0 else 0.0
        )
        over_tokens = token_ratio > 1.0
        over_latency = latency_ratio > 1.0

        # Score ramps from 0 at budget to 1.0 at three times budget. Prototype curve,
        # documented in the README as an assumption.
        worst_ratio = max(token_ratio, latency_ratio)
        score = min(1.0, max(0.0, (worst_ratio - 1.0) / 2.0))

        if score >= 0.67:
            severity = Severity.HIGH
        elif score >= 0.34:
            severity = Severity.MEDIUM
        elif score > 0.0:
            severity = Severity.LOW
        else:
            severity = Severity.NONE

        parts: list[str] = []
        if total is not None:
            parts.append(f"{total} tokens vs budget {token_budget} ({token_ratio:.2f}x)")
        else:
            parts.append("token usage not reported by provider")
        parts.append(
            f"{result.latency_ms:.0f} ms vs budget {latency_budget_ms:.0f} ms "
            f"({latency_ratio:.2f}x)"
        )
        if cost is not None:
            parts.append(f"estimated ${cost:.6f}")

        error = None if usage.complete else "provider did not report complete token usage"
        status = SignalStatus.WARN if (over_tokens or over_latency) else SignalStatus.PASS

        return CostSignal(
            status=status,
            score=score,
            severity=severity,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=total,
            estimated_cost_usd=cost,
            llm_latency_ms=result.latency_ms,
            over_token_budget=over_tokens,
            over_latency_budget=over_latency,
            explanation="; ".join(parts) + ".",
            error=error,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )


__all__ = ["CostService"]
