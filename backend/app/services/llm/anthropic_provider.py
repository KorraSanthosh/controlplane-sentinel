"""Anthropic provider adapter.

Two deliberate choices here:

* ``base_url`` is always passed explicitly. The SDK otherwise reads ambient
  ``ANTHROPIC_BASE_URL``, which is commonly set by other tooling — inheriting it would
  silently route this application's traffic through someone else's gateway.
* ``api_key`` is likewise passed explicitly from :class:`Settings`, never inherited.

Failures raise :class:`LLMUnavailable` rather than returning empty text, so a dead provider
can never be mistaken for a successful empty generation.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.config import Settings
from app.services.llm.base import (
    LLMProvider,
    LLMResult,
    LLMUnavailable,
    LLMUsage,
    ProviderHealth,
)

_JUDGE_SYSTEM = (
    "You are a strict content-safety reviewer for an enterprise support assistant. "
    "Reply with exactly one word: SAFE or UNSAFE, then a colon and a short reason."
)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise LLMUnavailable(
                "CP_ANTHROPIC_API_KEY is not set. Set it, or run with CP_LLM_PROVIDER=mock."
            )
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMUnavailable(
                "the `anthropic` package is not installed; `pip install -r requirements.txt`"
            ) from exc

        self.settings = settings
        self.model = settings.anthropic_model
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            base_url=settings.anthropic_base_url,
            timeout=settings.llm_timeout_s,
        )

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _extract_text(message: Any) -> str:
        """Concatenate text blocks, ignoring non-text content types."""
        parts: list[str] = []
        for block in getattr(message, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts).strip()

    def parse_usage(self, raw: Any) -> LLMUsage:
        """Read provider usage. Missing counts stay ``None`` — a silent 0 would
        understate cost risk rather than reporting it as unknown."""
        if raw is None:
            return LLMUsage()
        inp = getattr(raw, "input_tokens", None)
        out = getattr(raw, "output_tokens", None)
        if isinstance(raw, dict):
            inp = raw.get("input_tokens", inp)
            out = raw.get("output_tokens", out)
        total = None if inp is None or out is None else inp + out
        return LLMUsage(input_tokens=inp, output_tokens=out, total_tokens=total)

    async def _create(
        self,
        *,
        prompt: str,
        system: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature

        started = time.perf_counter()
        try:
            message = await self._client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalise every provider failure
            raise LLMUnavailable(f"Anthropic request failed: {type(exc).__name__}: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        return LLMResult(
            text=self._extract_text(message),
            provider=self.name,
            model=getattr(message, "model", self.model),
            usage=self.parse_usage(getattr(message, "usage", None)),
            latency_ms=latency_ms,
            stop_reason=getattr(message, "stop_reason", None),
            raw_metadata={
                "id": getattr(message, "id", None),
                "stop_sequence": getattr(message, "stop_sequence", None),
            },
        )

    # -- interface -----------------------------------------------------------
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResult:
        return await self._create(
            prompt=prompt, system=system, max_tokens=max_tokens, temperature=temperature
        )

    async def evaluate(
        self,
        instruction: str,
        content: str,
        *,
        max_tokens: int | None = None,
    ) -> LLMResult:
        return await self._create(
            prompt=f"{instruction}\n\n---\nContent to review:\n{content}",
            system=_JUDGE_SYSTEM,
            max_tokens=max_tokens or 128,
            temperature=0.0,
        )

    async def health_check(self) -> ProviderHealth:
        """Cheap reachability probe — lists models rather than spending output tokens."""
        try:
            await self._client.models.list(limit=1)
            return ProviderHealth(
                name=self.name, healthy=True, model=self.model, detail="API reachable"
            )
        except Exception as exc:  # noqa: BLE001
            return ProviderHealth(
                name=self.name,
                healthy=False,
                model=self.model,
                detail=f"{type(exc).__name__}: {exc}",
            )

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            await close()


__all__ = ["AnthropicProvider"]
