"""The declarative condition language used by policy rules.

Policy rules are data, not code (SYSTEM_REQUIREMENTS FR-09: thresholds must be
configuration-driven). That trade needs one safeguard: a typo in a field name must fail at
**load time**, not turn into a rule that silently never matches. A rule that never fires and a
rule that always passes look identical from the outside, and the second one is a governance
hole. So every field name, operator and operand is validated and coerced when the profile is
read, and :class:`Condition` objects are only ever built from validated input.

Field reference (see ``FIELDS`` for the authoritative list):

    grounding.status              grounded | unsupported | contradicted | unavailable
    grounding.signal_status       pass | warn | fail | unavailable | skipped
    grounding.severity            none | low | medium | high | critical
    grounding.score               0.0 .. 1.0
    grounding.claims_checked      int
    grounding.claims_contradicted int
    grounding.claims_unsupported  int
    grounding.claims_unverifiable int
    pii.status / pii.severity / pii.score / pii.detected / pii.redactable
    pii.match_count               int
    pii.kinds                     list of PII kind names
    safety.status / safety.severity / safety.score / safety.judge_used
    safety.violation_count        int
    safety.categories             list of violated policy category names
    cost.status / cost.score / cost.over_token_budget / cost.over_latency_budget
    cost.total_tokens             int or null when the provider did not report usage
    cost.latency_ms               float
    cost.estimated_cost_usd       float or null
    risk.overall_score            0.0 .. 1.0
    risk.unavailable_checks       list of detector names
    risk.skipped_checks           list of detector names
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from app.schemas.signals import (
    GroundingStatus,
    RiskAssessment,
    Severity,
    SignalStatus,
    severity_rank,
)


class PolicyError(ValueError):
    """A policy profile is malformed.

    Always raised while loading, never while deciding. A live request must never fail because
    of a policy typo that could have been caught at startup.
    """


# ---------------------------------------------------------------------------
# Field registry
# ---------------------------------------------------------------------------
#: Value types drive both operand coercion and which operators are legal.
SEVERITY = "severity"
SIGNAL_STATUS = "signal_status"
GROUNDING_STATUS = "grounding_status"
NUMBER = "number"
BOOL = "bool"
STR_LIST = "str_list"


@dataclass(frozen=True)
class FieldSpec:
    resolve: Callable[[RiskAssessment], Any]
    value_type: str


FIELDS: dict[str, FieldSpec] = {
    # -- grounding ---------------------------------------------------------
    "grounding.status": FieldSpec(
        lambda a: a.signals.grounding.grounding_status, GROUNDING_STATUS
    ),
    "grounding.signal_status": FieldSpec(lambda a: a.signals.grounding.status, SIGNAL_STATUS),
    "grounding.severity": FieldSpec(lambda a: a.signals.grounding.severity, SEVERITY),
    "grounding.score": FieldSpec(lambda a: a.signals.grounding.score, NUMBER),
    "grounding.claims_checked": FieldSpec(lambda a: a.signals.grounding.claims_checked, NUMBER),
    "grounding.claims_contradicted": FieldSpec(
        lambda a: a.signals.grounding.claims_contradicted, NUMBER
    ),
    "grounding.claims_unsupported": FieldSpec(
        lambda a: a.signals.grounding.claims_unsupported, NUMBER
    ),
    "grounding.claims_unverifiable": FieldSpec(
        lambda a: a.signals.grounding.claims_unverifiable, NUMBER
    ),
    # -- pii ---------------------------------------------------------------
    "pii.status": FieldSpec(lambda a: a.signals.pii.status, SIGNAL_STATUS),
    "pii.severity": FieldSpec(lambda a: a.signals.pii.severity, SEVERITY),
    "pii.score": FieldSpec(lambda a: a.signals.pii.score, NUMBER),
    "pii.detected": FieldSpec(lambda a: a.signals.pii.detected, BOOL),
    "pii.redactable": FieldSpec(lambda a: a.signals.pii.redactable, BOOL),
    "pii.match_count": FieldSpec(lambda a: len(a.signals.pii.matches), NUMBER),
    "pii.kinds": FieldSpec(lambda a: sorted(a.signals.pii.counts), STR_LIST),
    # -- safety ------------------------------------------------------------
    "safety.status": FieldSpec(lambda a: a.signals.safety.status, SIGNAL_STATUS),
    "safety.severity": FieldSpec(lambda a: a.signals.safety.severity, SEVERITY),
    "safety.score": FieldSpec(lambda a: a.signals.safety.score, NUMBER),
    "safety.judge_used": FieldSpec(lambda a: a.signals.safety.judge_used, BOOL),
    "safety.violation_count": FieldSpec(lambda a: len(a.signals.safety.violations), NUMBER),
    "safety.categories": FieldSpec(
        lambda a: sorted({v.category for v in a.signals.safety.violations}), STR_LIST
    ),
    # -- cost --------------------------------------------------------------
    "cost.status": FieldSpec(lambda a: a.signals.cost.status, SIGNAL_STATUS),
    "cost.score": FieldSpec(lambda a: a.signals.cost.score, NUMBER),
    "cost.over_token_budget": FieldSpec(lambda a: a.signals.cost.over_token_budget, BOOL),
    "cost.over_latency_budget": FieldSpec(lambda a: a.signals.cost.over_latency_budget, BOOL),
    "cost.total_tokens": FieldSpec(lambda a: a.signals.cost.total_tokens, NUMBER),
    "cost.latency_ms": FieldSpec(lambda a: a.signals.cost.llm_latency_ms, NUMBER),
    "cost.estimated_cost_usd": FieldSpec(lambda a: a.signals.cost.estimated_cost_usd, NUMBER),
    # -- aggregate ---------------------------------------------------------
    "risk.overall_score": FieldSpec(lambda a: a.overall_score, NUMBER),
    "risk.unavailable_checks": FieldSpec(lambda a: sorted(a.unavailable_checks), STR_LIST),
    "risk.skipped_checks": FieldSpec(lambda a: sorted(a.skipped_checks), STR_LIST),
}


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
_EQUALITY = frozenset({"eq", "ne"})
_ORDERED = frozenset({"gt", "gte", "lt", "lte"})
_MEMBERSHIP = frozenset({"in", "not_in"})
_SET_OPS = frozenset({"any_in", "none_in"})

#: Which operators make sense per value type. ``pass``/``warn``/``fail`` have no meaningful
#: ordering, so ``gte`` is rejected for status fields — writing ``status: {gte: warn}`` would
#: read as if it worked while comparing enum declaration order by accident.
ALLOWED_OPERATORS: dict[str, frozenset[str]] = {
    SEVERITY: _EQUALITY | _ORDERED | _MEMBERSHIP,
    SIGNAL_STATUS: _EQUALITY | _MEMBERSHIP,
    GROUNDING_STATUS: _EQUALITY | _MEMBERSHIP,
    NUMBER: _EQUALITY | _ORDERED,
    BOOL: _EQUALITY,
    STR_LIST: _SET_OPS,
}


def _coerce_scalar(value_type: str, raw: Any, where: str) -> Any:
    """Turn a YAML scalar into the field's native type, or fail loudly."""
    try:
        if value_type == SEVERITY:
            return Severity(str(raw).strip().lower())
        if value_type == SIGNAL_STATUS:
            return SignalStatus(str(raw).strip().lower())
        if value_type == GROUNDING_STATUS:
            return GroundingStatus(str(raw).strip().lower())
        if value_type == NUMBER:
            return float(raw)
        if value_type == BOOL:
            if isinstance(raw, bool):
                return raw
            token = str(raw).strip().lower()
            if token in {"true", "yes", "1"}:
                return True
            if token in {"false", "no", "0"}:
                return False
            raise ValueError(f"{raw!r} is not a boolean")
        return str(raw)
    except (ValueError, TypeError) as exc:
        raise PolicyError(f"{where}: invalid {value_type} operand {raw!r} ({exc})") from exc


def _coerce_operand(value_type: str, operator: str, raw: Any, where: str) -> Any:
    if operator in _MEMBERSHIP | _SET_OPS:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise PolicyError(f"{where}: operator '{operator}' needs a list, got {raw!r}")
        return tuple(_coerce_scalar(value_type, item, where) for item in raw)
    return _coerce_scalar(value_type, raw, where)


def _rank(value_type: str, value: Any) -> float:
    """Project a value onto a number line for ordered comparison."""
    if value_type == SEVERITY:
        return float(severity_rank(value))
    return float(value)


def _display(value: Any) -> Any:
    """JSON-friendly rendering for the audit trail."""
    if isinstance(value, (Severity, SignalStatus, GroundingStatus)):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_display(v) for v in value]
    return value


@dataclass(frozen=True)
class Condition:
    """One validated ``field operator operand`` test."""

    field: str
    operator: str
    operand: Any
    value_type: str

    def evaluate(self, assessment: RiskAssessment) -> tuple[bool, dict[str, Any]]:
        """Return ``(matched, trace)``. The trace is recorded on the fired rule."""
        actual = FIELDS[self.field].resolve(assessment)
        matched = self._compare(actual)
        return matched, {
            "operator": self.operator,
            "expected": _display(self.operand),
            "actual": _display(actual),
        }

    def _compare(self, actual: Any) -> bool:
        op = self.operator

        if self.value_type == STR_LIST:
            present = {str(v) for v in (actual or ())}
            wanted = {str(v) for v in self.operand}
            if op == "any_in":
                return bool(present & wanted)
            return not (present & wanted)  # none_in

        # A field the provider never reported (e.g. token counts) is unknown, not zero.
        # "Unknown" must not satisfy a budget-overrun test.
        if actual is None:
            return False

        if op == "eq":
            return actual == self.operand
        if op == "ne":
            return actual != self.operand
        if op == "in":
            return actual in self.operand
        if op == "not_in":
            return actual not in self.operand

        left, right = _rank(self.value_type, actual), _rank(self.value_type, self.operand)
        if op == "gt":
            return left > right
        if op == "gte":
            return left >= right
        if op == "lt":
            return left < right
        return left <= right  # lte


def build_conditions(raw: Any, where: str) -> tuple[Condition, ...]:
    """Compile a rule's ``when:`` block into validated conditions.

    Two accepted spellings per entry::

        pii.detected: true                  # shorthand for {eq: true}
        safety.severity: {gte: critical}    # explicit operator

    An empty or missing ``when`` is rejected: a rule that matches everything belongs in the
    profile's ``default_action``, where it is visible, not hidden among the rules.
    """
    if not raw:
        raise PolicyError(f"{where}: 'when' must contain at least one condition")
    if not isinstance(raw, dict):
        raise PolicyError(f"{where}: 'when' must be a mapping of field -> test")

    conditions: list[Condition] = []
    for field, test in raw.items():
        spec = FIELDS.get(field)
        if spec is None:
            raise PolicyError(
                f"{where}: unknown condition field '{field}'. "
                f"Known fields: {', '.join(sorted(FIELDS))}"
            )

        if isinstance(test, dict):
            if len(test) != 1:
                raise PolicyError(
                    f"{where}.{field}: expected exactly one operator, got {sorted(test)}"
                )
            operator, operand = next(iter(test.items()))
        else:
            operator, operand = "eq", test

        operator = str(operator).strip().lower()
        allowed = ALLOWED_OPERATORS[spec.value_type]
        if operator not in allowed:
            raise PolicyError(
                f"{where}.{field}: operator '{operator}' is not valid for a "
                f"{spec.value_type} field. Allowed: {', '.join(sorted(allowed))}"
            )

        conditions.append(
            Condition(
                field=field,
                operator=operator,
                operand=_coerce_operand(spec.value_type, operator, operand, f"{where}.{field}"),
                value_type=spec.value_type,
            )
        )

    return tuple(conditions)


__all__ = [
    "ALLOWED_OPERATORS",
    "FIELDS",
    "Condition",
    "FieldSpec",
    "PolicyError",
    "build_conditions",
]
