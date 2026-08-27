"""Provider selection.

The only place in the codebase that knows which vendor is in play. Everything downstream
depends on :class:`LLMProvider` alone.
"""

from __future__ import annotations

import logging

from app.core.config import Settings
from app.demo.scenarios import cached_scenario_library
from app.services.llm.base import LLMProvider, LLMUnavailable
from app.services.llm.mock_provider import MockProvider

logger = logging.getLogger(__name__)


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Construct the configured provider.

    If ``anthropic`` is requested but unusable, we fall back to the mock provider and log a
    warning rather than refusing to start — a demo box with no key should still boot and
    serve every scenario. ``/health`` reports the provider actually in use, so the fallback
    is visible rather than silent.
    """
    library = cached_scenario_library(str(settings.scenarios_path_abs))

    if settings.llm_provider == "anthropic":
        try:
            from app.services.llm.anthropic_provider import AnthropicProvider

            return AnthropicProvider(settings)
        except LLMUnavailable as exc:
            logger.warning(
                "anthropic provider unavailable (%s); falling back to mock provider", exc
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "anthropic provider failed to initialise (%s: %s); falling back to mock",
                type(exc).__name__,
                exc,
            )

    return MockProvider(library)


__all__ = ["build_llm_provider"]
