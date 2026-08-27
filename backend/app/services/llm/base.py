"""LLM provider abstraction.

Nothing provider-specific may leak past this module: the orchestrator, detectors and API
layer only ever see :class:`LLMResult`. Swapping Anthropic for another vendor means adding
one file here and changing ``CP_LLM_PROVIDER`` — no changes to business logic.
"""

from __future__ import annotations

import abc
from typing import Any

from pydantic import BaseModel, Field


class LLMUsage(BaseModel):
    """Token accounting. Every field is optional because not all providers report it,
    and a missing count must stay visibly missing rather than defaulting to 0 — a silent
    0 would understate cost risk."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def complete(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None


class LLMResult(BaseModel):
    text: str
    provider: str
    model: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    latency_ms: float = 0.0
    stop_reason: str | None = None
    #: Non-sensitive provider metadata kept for telemetry. Never store credentials here.
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderHealth(BaseModel):
    name: str
    healthy: bool
    detail: str = ""
    model: str | None = None


class LLMError(Exception):
    """Base class for provider failures."""


class LLMUnavailable(LLMError):
    """The provider could not be reached or is misconfigured.

    Raised rather than returning empty text, so the caller cannot mistake a failed
    generation for a successful empty one.
    """


class LLMProvider(abc.ABC):
    """The interface required by CLAUDE.md §7."""

    name: str = "base"

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        """Produce a response to a user prompt."""

    @abc.abstractmethod
    async def evaluate(
        self,
        instruction: str,
        content: str,
        *,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Judge or classify existing content (the AI-as-judge deep path).

        Separate from :meth:`generate` because it is a distinct cost centre and a distinct
        failure mode: a judge outage must degrade the safety check to ``unavailable``
        without touching the primary generation path.
        """

    @abc.abstractmethod
    async def health_check(self) -> ProviderHealth:
        """Report reachability without incurring a full generation where possible."""

    def parse_usage(self, raw: Any) -> LLMUsage:  # pragma: no cover - overridden
        """Translate provider-native usage metadata into :class:`LLMUsage`."""
        return LLMUsage()

    async def aclose(self) -> None:
        """Release connections. Default is a no-op."""
        return None


__all__ = [
    "LLMError",
    "LLMProvider",
    "LLMResult",
    "LLMUnavailable",
    "LLMUsage",
    "ProviderHealth",
]
