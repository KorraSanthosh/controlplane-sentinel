"""Application configuration.

All settings are read from environment variables (or a local `.env`) with the ``CP_``
prefix. The prefix is deliberate: ambient ``ANTHROPIC_*`` variables are commonly set by
other tooling, and the Anthropic SDK auto-reads ``ANTHROPIC_API_KEY`` /
``ANTHROPIC_BASE_URL``. Prefixing everything means this app can never silently inherit
another tool's credentials or gateway.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CP_",
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application ---------------------------------------------------------
    app_name: str = "ControlPlane.ai"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"

    # --- LLM ----------------------------------------------------------------
    llm_provider: Literal["mock", "anthropic"] = "mock"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5"
    # Explicit rather than SDK-default so ambient ANTHROPIC_BASE_URL cannot redirect us.
    anthropic_base_url: str = "https://api.anthropic.com"
    llm_max_tokens: int = 1024
    llm_timeout_s: float = 30.0

    # --- Graph (Neo4j Aura) -------------------------------------------------
    neo4j_uri: str | None = None
    neo4j_user: str = "neo4j"
    neo4j_password: str | None = None
    neo4j_database: str = "neo4j"

    # --- Document store (MongoDB Atlas) -------------------------------------
    mongo_uri: str | None = None
    mongo_db: str = "controlplane"

    # --- Policy -------------------------------------------------------------
    policy_dir: Path = Path("policies")
    default_policy_profile: str = "default"

    # --- Demo fixtures ------------------------------------------------------
    scenarios_path: Path = Path("data/demo/scenarios.yaml")
    graph_seed_path: Path = Path("graph/seed/northwind.json")

    # --- Cost estimation ----------------------------------------------------
    # PROTOTYPE ASSUMPTION, not authoritative pricing. See README.
    price_input_per_mtok: float = 3.00
    price_output_per_mtok: float = 15.00

    # --- Debug --------------------------------------------------------------
    # When true, /chat may echo the original (pre-action) model text so the dashboard can
    # show "original vs delivered". Must stay false in any real deployment: it would
    # return content the policy engine decided to withhold.
    allow_debug_original: bool = Field(
        default=True,
        description="Permit ?debug=true on /chat to reveal the pre-action model output.",
    )

    # --- Derived paths ------------------------------------------------------
    @property
    def policy_dir_abs(self) -> Path:
        return self._abs(self.policy_dir)

    @property
    def scenarios_path_abs(self) -> Path:
        return self._abs(self.scenarios_path)

    @property
    def graph_seed_path_abs(self) -> Path:
        return self._abs(self.graph_seed_path)

    @staticmethod
    def _abs(p: Path) -> Path:
        return p if p.is_absolute() else (REPO_ROOT / p)

    # --- Capability checks --------------------------------------------------
    @property
    def neo4j_configured(self) -> bool:
        return bool(self.neo4j_uri and self.neo4j_password)

    @property
    def mongo_configured(self) -> bool:
        return bool(self.mongo_uri)

    @property
    def anthropic_configured(self) -> bool:
        return bool(self.anthropic_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
