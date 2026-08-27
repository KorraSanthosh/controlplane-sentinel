"""Policy profile loading and validation.

A *profile* is one YAML file describing everything tunable about how ControlPlane behaves:
which checks run, how their scores are weighted, what the cost budgets are, when the expensive
deep path is worth paying for, what to do when a detector is unavailable, and the ordered
decision rules. Three ship with the prototype — ``default``, ``strict``, ``lenient`` — so the
same response can be shown producing different actions under different governance postures.

Everything here is validated eagerly. ``load_profile`` raises :class:`PolicyError` on the first
problem it finds, and the application refuses to start rather than serve traffic under a policy
it could not fully parse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.schemas.decision import Decision
from app.schemas.signals import GroundingStatus, SignalStatus
from app.services.policy.conditions import Condition, PolicyError, build_conditions

logger = logging.getLogger(__name__)

#: Files in the policy directory that are not profiles. ``safety_rules.yaml`` is the shared
#: rule corpus consumed by SafetyService, not a governance profile.
NON_PROFILE_FILES = frozenset({"safety_rules.yaml"})

#: How to resolve several matching rules.
STRATEGY_MOST_RESTRICTIVE = "most_restrictive"
STRATEGY_FIRST_MATCH = "first_match"
STRATEGIES = frozenset({STRATEGY_MOST_RESTRICTIVE, STRATEGY_FIRST_MATCH})

CHECK_NAMES = ("grounding", "pii", "safety", "cost")


@dataclass(frozen=True)
class Budget:
    """Per-use-case cost ceiling. Prototype values, not industry benchmarks."""

    max_total_tokens: int
    max_latency_ms: float


@dataclass(frozen=True)
class TriageConfig:
    """When the deep path is worth paying for.

    This is the concrete answer to the brief's latency question. The fast path (regex PII,
    rule-based safety, cost arithmetic, a regex claim pre-scan) runs on every request and costs
    single-digit milliseconds. The two expensive operations — the graph fact query and the LLM
    judge — run only when one of these gates opens.
    """

    #: Query the graph only when the pre-scan actually extracted a verifiable claim. A response
    #: that asserts nothing checkable never pays for graph I/O.
    deep_grounding_on_claims: bool = True
    #: Skip grounding verification entirely, even when claims exist. Escape hatch for
    #: latency-critical use cases.
    deep_grounding_enabled: bool = True
    #: Run the LLM judge only when a cheap signal is already concerned.
    judge_enabled: bool = True
    judge_on_fast_risk_at_or_above: float = 0.35
    judge_on_grounding_status: frozenset[GroundingStatus] = frozenset()
    judge_on_safety_status: frozenset[SignalStatus] = frozenset()


@dataclass(frozen=True)
class DecisionRule:
    id: str
    action: Decision
    reason: str
    conditions: tuple[Condition, ...]
    requires_human_review: bool = False
    #: Set on PII rules so masking is applied even when a higher tier wins the decision.
    apply_redaction: bool = False


@dataclass(frozen=True)
class PolicyProfile:
    id: str
    version: str
    title: str
    description: str
    enabled_checks: dict[str, bool]
    weights: dict[str, float]
    default_budget: Budget
    budgets: dict[str, Budget]
    triage: TriageConfig
    on_unavailable: dict[str, Decision]
    strategy: str
    default_action: Decision
    default_reason: str
    rules: tuple[DecisionRule, ...]
    #: Delivered in place of the model output on BLOCK. Governance-owned user-facing copy, so
    #: it lives in the profile rather than being hardcoded in the pipeline.
    blocked_response: str = (
        "I'm not able to share that response. It has been withheld by an automated "
        "responsible-AI check and routed to a human for review. Please rephrase your "
        "question, or ask to be transferred to a support agent."
    )
    #: None means "every category in safety_rules.yaml is in force".
    safety_categories: tuple[str, ...] | None = None
    source_path: str = ""

    def budget_for(self, use_case: str | None) -> Budget:
        return self.budgets.get(use_case or "", self.default_budget)

    def is_enabled(self, check: str) -> bool:
        return bool(self.enabled_checks.get(check, True))

    def unavailable_action(self, check: str) -> Decision:
        """What to do when ``check`` could not run. Defaults to FLAG, never ALLOW.

        FR-11: an unavailable detector must not resolve to a silent pass. If a profile wants
        that (``cost`` is a reasonable case — a missing token count is not a safety matter) it
        has to say ALLOW out loud.
        """
        return self.on_unavailable.get(check, Decision.FLAG)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _require_mapping(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PolicyError(f"{where}: expected a mapping, got {type(raw).__name__}")
    return raw


def _decision(raw: Any, where: str) -> Decision:
    try:
        return Decision(str(raw).strip().upper())
    except ValueError as exc:
        raise PolicyError(
            f"{where}: '{raw}' is not a decision. "
            f"Expected one of {', '.join(d.value for d in Decision)}"
        ) from exc


def _parse_budget(raw: Any, where: str) -> Budget:
    data = _require_mapping(raw, where)
    try:
        tokens = int(data["max_total_tokens"])
        latency = float(data["max_latency_ms"])
    except KeyError as exc:
        raise PolicyError(f"{where}: missing required key {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise PolicyError(f"{where}: non-numeric budget value ({exc})") from exc
    if tokens <= 0 or latency <= 0:
        raise PolicyError(f"{where}: budgets must be positive")
    return Budget(max_total_tokens=tokens, max_latency_ms=latency)


def _parse_weights(raw: Any, where: str) -> dict[str, float]:
    data = _require_mapping(raw or {}, where)
    weights: dict[str, float] = {}
    for name, value in data.items():
        if name not in CHECK_NAMES:
            raise PolicyError(
                f"{where}: unknown check '{name}'. Known: {', '.join(CHECK_NAMES)}"
            )
        try:
            weight = float(value)
        except (TypeError, ValueError) as exc:
            raise PolicyError(f"{where}.{name}: non-numeric weight {value!r}") from exc
        if weight < 0:
            raise PolicyError(f"{where}.{name}: weight must not be negative")
        weights[name] = weight
    if not weights:
        raise PolicyError(f"{where}: at least one weight is required")
    if sum(weights.values()) <= 0:
        raise PolicyError(f"{where}: weights sum to zero, no signal could ever score")
    return weights


def _parse_triage(raw: Any, where: str) -> TriageConfig:
    data = _require_mapping(raw or {}, where)

    def _statuses(key: str, enum: type, default: tuple[str, ...] = ()) -> frozenset:
        values = data.get(key, list(default))
        if isinstance(values, str) or not isinstance(values, (list, tuple)):
            raise PolicyError(f"{where}.{key}: expected a list")
        out = []
        for item in values:
            try:
                out.append(enum(str(item).strip().lower()))
            except ValueError as exc:
                raise PolicyError(f"{where}.{key}: invalid value '{item}'") from exc
        return frozenset(out)

    try:
        threshold = float(data.get("judge_on_fast_risk_at_or_above", 0.35))
    except (TypeError, ValueError) as exc:
        raise PolicyError(f"{where}.judge_on_fast_risk_at_or_above: not a number") from exc
    if not 0.0 <= threshold <= 1.0:
        raise PolicyError(f"{where}.judge_on_fast_risk_at_or_above: must be within 0.0..1.0")

    return TriageConfig(
        deep_grounding_on_claims=bool(data.get("deep_grounding_on_claims", True)),
        deep_grounding_enabled=bool(data.get("deep_grounding_enabled", True)),
        judge_enabled=bool(data.get("judge_enabled", True)),
        judge_on_fast_risk_at_or_above=threshold,
        judge_on_grounding_status=_statuses("judge_on_grounding_status", GroundingStatus),
        judge_on_safety_status=_statuses("judge_on_safety_status", SignalStatus),
    )


def _parse_rules(raw: Any, where: str) -> tuple[DecisionRule, ...]:
    if not isinstance(raw, list) or not raw:
        raise PolicyError(f"{where}: at least one decision rule is required")

    rules: list[DecisionRule] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        data = _require_mapping(item, f"{where}[{index}]")
        rule_id = str(data.get("id") or "").strip()
        if not rule_id:
            raise PolicyError(f"{where}[{index}]: 'id' is required (it is written to the audit)")
        if rule_id in seen:
            raise PolicyError(f"{where}[{index}]: duplicate rule id '{rule_id}'")
        seen.add(rule_id)

        location = f"{where}[{rule_id}]"
        reason = str(data.get("reason") or "").strip()
        if not reason:
            raise PolicyError(f"{location}: 'reason' is required — a decision must be explainable")

        rules.append(
            DecisionRule(
                id=rule_id,
                action=_decision(data.get("action"), f"{location}.action"),
                reason=" ".join(reason.split()),
                conditions=build_conditions(data.get("when"), location),
                requires_human_review=bool(data.get("requires_human_review", False)),
                apply_redaction=bool(data.get("apply_redaction", False)),
            )
        )
    return tuple(rules)


def load_profile(path: Path) -> PolicyProfile:
    """Read and fully validate one profile file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise PolicyError(f"{path}: cannot be read ({exc})") from exc
    except yaml.YAMLError as exc:
        raise PolicyError(f"{path}: invalid YAML ({exc})") from exc

    data = _require_mapping(raw, str(path))
    profile_id = str(data.get("id") or path.stem).strip()
    where = f"{path.name}"

    enabled_raw = _require_mapping(data.get("enabled_checks") or {}, f"{where}.enabled_checks")
    for name in enabled_raw:
        if name not in CHECK_NAMES:
            raise PolicyError(
                f"{where}.enabled_checks: unknown check '{name}'. "
                f"Known: {', '.join(CHECK_NAMES)}"
            )
    enabled = {name: bool(enabled_raw.get(name, True)) for name in CHECK_NAMES}

    budgets_raw = _require_mapping(data.get("budgets") or {}, f"{where}.budgets")
    if "default" not in budgets_raw:
        raise PolicyError(f"{where}.budgets: a 'default' budget is required")
    default_budget = _parse_budget(budgets_raw["default"], f"{where}.budgets.default")
    budgets = {
        name: _parse_budget(value, f"{where}.budgets.{name}")
        for name, value in budgets_raw.items()
        if name != "default"
    }

    on_unavailable_raw = _require_mapping(
        data.get("on_unavailable") or {}, f"{where}.on_unavailable"
    )
    for name in on_unavailable_raw:
        if name not in CHECK_NAMES:
            raise PolicyError(f"{where}.on_unavailable: unknown check '{name}'")
    on_unavailable = {
        name: _decision(value, f"{where}.on_unavailable.{name}")
        for name, value in on_unavailable_raw.items()
    }

    strategy = str(data.get("strategy") or STRATEGY_MOST_RESTRICTIVE).strip().lower()
    if strategy not in STRATEGIES:
        raise PolicyError(
            f"{where}.strategy: '{strategy}' is not valid. Expected one of "
            f"{', '.join(sorted(STRATEGIES))}"
        )

    categories_raw = data.get("safety_categories")
    if categories_raw is None:
        categories: tuple[str, ...] | None = None
    elif isinstance(categories_raw, list):
        categories = tuple(str(c).strip() for c in categories_raw if str(c).strip())
    else:
        raise PolicyError(f"{where}.safety_categories: expected a list or omission")

    default_reason = " ".join(
        str(
            data.get("default_reason")
            or "No policy rule matched; the response was delivered unchanged."
        ).split()
    )

    blocked_raw = data.get("blocked_response")
    blocked_kwargs = (
        {"blocked_response": " ".join(str(blocked_raw).split())} if blocked_raw else {}
    )

    return PolicyProfile(
        id=profile_id,
        version=str(data.get("version") or "0"),
        title=str(data.get("title") or profile_id),
        description=" ".join(str(data.get("description") or "").split()),
        enabled_checks=enabled,
        weights=_parse_weights(data.get("weights"), f"{where}.weights"),
        default_budget=default_budget,
        budgets=budgets,
        triage=_parse_triage(data.get("triage"), f"{where}.triage"),
        on_unavailable=on_unavailable,
        strategy=strategy,
        default_action=_decision(data.get("default_action") or "ALLOW", f"{where}.default_action"),
        default_reason=default_reason,
        rules=_parse_rules(data.get("rules"), f"{where}.rules"),
        safety_categories=categories,
        source_path=str(path),
        **blocked_kwargs,
    )


@dataclass
class PolicyRegistry:
    profiles: dict[str, PolicyProfile] = field(default_factory=dict)
    default_id: str = "default"

    def get(self, profile_id: str | None = None) -> PolicyProfile:
        """Resolve a profile by id, falling back to the configured default.

        An unknown id falls back with a warning rather than raising: a bad ``policy_profile``
        in a request body should not 500, and quietly running *no* policy is not an option.
        """
        wanted = (profile_id or self.default_id).strip()
        profile = self.profiles.get(wanted)
        if profile is not None:
            return profile
        if profile_id:
            logger.warning(
                "Unknown policy profile %r; falling back to %r", profile_id, self.default_id
            )
        fallback = self.profiles.get(self.default_id)
        if fallback is None:
            raise PolicyError(
                f"No policy profile '{self.default_id}' is loaded and none was requested"
            )
        return fallback

    def ids(self) -> list[str]:
        return sorted(self.profiles)


def load_policy_registry(policy_dir: Path, default_id: str = "default") -> PolicyRegistry:
    if not policy_dir.is_dir():
        raise PolicyError(f"Policy directory not found: {policy_dir}")

    paths = sorted(
        p
        for p in policy_dir.glob("*.y*ml")
        if p.is_file() and p.name not in NON_PROFILE_FILES
    )
    if not paths:
        raise PolicyError(f"No policy profiles found in {policy_dir}")

    profiles: dict[str, PolicyProfile] = {}
    for path in paths:
        profile = load_profile(path)
        if profile.id in profiles:
            raise PolicyError(
                f"{path.name}: duplicate profile id '{profile.id}' "
                f"(already defined by {profiles[profile.id].source_path})"
            )
        profiles[profile.id] = profile

    if default_id not in profiles:
        raise PolicyError(
            f"Default policy profile '{default_id}' not found. Loaded: {', '.join(profiles)}"
        )

    logger.info("Loaded %d policy profile(s): %s", len(profiles), ", ".join(sorted(profiles)))
    return PolicyRegistry(profiles=profiles, default_id=default_id)


@lru_cache(maxsize=8)
def cached_policy_registry(policy_dir: Path, default_id: str) -> PolicyRegistry:
    return load_policy_registry(policy_dir, default_id)


__all__ = [
    "CHECK_NAMES",
    "NON_PROFILE_FILES",
    "STRATEGIES",
    "STRATEGY_FIRST_MATCH",
    "STRATEGY_MOST_RESTRICTIVE",
    "Budget",
    "DecisionRule",
    "PolicyError",
    "PolicyProfile",
    "PolicyRegistry",
    "TriageConfig",
    "cached_policy_registry",
    "load_policy_registry",
    "load_profile",
]
