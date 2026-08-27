"""Safety and policy checking.

Two layers, deliberately separated:

* **Rule layer** (fast path) — deterministic phrase/regex rules from
  ``policies/safety_rules.yaml``. Runs inline on every response. Cheap, explainable, and
  every hit names a policy id.
* **Judge layer** (deep path) — an LLM second opinion, run only when triage asks for it.

The rule layer is authoritative for the *status*. The judge can escalate severity and adds
its own rationale, but a judge failure never downgrades a rule finding, and a judge outage
is recorded as ``unavailable`` on that sub-check rather than being treated as agreement.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from app.schemas.signals import (
    PolicyViolation,
    SafetySignal,
    Severity,
    SignalStatus,
    severity_rank,
)
from app.services.llm.base import LLMProvider

logger = logging.getLogger(__name__)

_JUDGE_INSTRUCTION = (
    "Review the following customer-support response for policy violations: unauthorised "
    "financial commitments, disclosure of confidential internal data, credential leakage, "
    "out-of-scope medical/legal/financial advice, competitor disparagement, or abusive "
    "language. Answer SAFE or UNSAFE with a one-line reason."
)

#: Risk score assigned per severity. Prototype policy, documented in the README — not an
#: industry standard.
_SEVERITY_SCORE = {
    Severity.NONE: 0.0,
    Severity.LOW: 0.25,
    Severity.MEDIUM: 0.50,
    Severity.HIGH: 0.75,
    Severity.CRITICAL: 1.0,
}


@dataclass(frozen=True)
class SafetyRule:
    id: str
    category: str
    severity: Severity
    reason: str
    phrases: tuple[str, ...]
    patterns: tuple[re.Pattern[str], ...]

    def find(self, text: str) -> str | None:
        """Return the matched excerpt, or None."""
        lowered = text.lower()
        for phrase in self.phrases:
            idx = lowered.find(phrase.lower())
            if idx != -1:
                return text[idx : idx + len(phrase)]
        for pat in self.patterns:
            m = pat.search(text)
            if m:
                return m.group(0)
        return None


def load_safety_rules(path: str | Path) -> tuple[SafetyRule, ...]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"safety rules not found at {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    rules: list[SafetyRule] = []
    for entry in raw.get("rules", []):
        rules.append(
            SafetyRule(
                id=entry["id"],
                category=entry.get("category", "unspecified"),
                severity=Severity(entry.get("severity", "medium")),
                reason=re.sub(r"\s+", " ", entry.get("reason", "")).strip(),
                phrases=tuple(entry.get("phrases") or ()),
                patterns=tuple(
                    re.compile(p, re.IGNORECASE) for p in (entry.get("patterns") or ())
                ),
            )
        )
    return tuple(rules)


@lru_cache(maxsize=4)
def cached_safety_rules(path: str) -> tuple[SafetyRule, ...]:
    return load_safety_rules(path)


class SafetyService:
    name = "safety"

    def __init__(self, rules: tuple[SafetyRule, ...], llm: LLMProvider | None = None) -> None:
        self.rules = rules
        self.llm = llm

    # -- fast path -----------------------------------------------------------
    def check_rules(
        self, text: str, *, enabled_categories: set[str] | None = None
    ) -> SafetySignal:
        started = time.perf_counter()
        violations: list[PolicyViolation] = []

        for rule in self.rules:
            if enabled_categories is not None and rule.category not in enabled_categories:
                continue
            excerpt = rule.find(text or "")
            if excerpt is None:
                continue
            violations.append(
                PolicyViolation(
                    policy_id=rule.id,
                    category=rule.category,
                    severity=rule.severity,
                    reason=rule.reason,
                    matched_preview=excerpt[:120],
                )
            )

        duration_ms = (time.perf_counter() - started) * 1000.0

        if not violations:
            return SafetySignal(
                status=SignalStatus.PASS,
                score=0.0,
                severity=Severity.NONE,
                explanation=f"No safety rule matched ({len(self.rules)} rules evaluated).",
                duration_ms=duration_ms,
            )

        peak = max((v.severity for v in violations), key=severity_rank)
        ids = ", ".join(sorted({v.policy_id for v in violations}))
        return SafetySignal(
            status=SignalStatus.FAIL,
            score=_SEVERITY_SCORE[peak],
            severity=peak,
            violations=violations,
            explanation=f"{len(violations)} policy violation(s): {ids}.",
            duration_ms=duration_ms,
        )

    # -- deep path -----------------------------------------------------------
    async def judge(self, text: str, base: SafetySignal) -> SafetySignal:
        """Add an LLM second opinion to an existing rule verdict.

        Returns a copy of ``base``. On judge failure the copy keeps the rule verdict and
        records the error — the response is never upgraded to PASS because the judge
        could not be reached.
        """
        if self.llm is None:
            return base.model_copy(
                update={
                    "judge_used": False,
                    "error": _join_errors(base.error, "no LLM provider configured for judge"),
                }
            )

        started = time.perf_counter()
        try:
            result = await self.llm.evaluate(_JUDGE_INSTRUCTION, text or "")
        except Exception as exc:  # noqa: BLE001 - judge must never break the request
            logger.warning("safety judge unavailable: %s: %s", type(exc).__name__, exc)
            return base.model_copy(
                update={
                    "judge_used": False,
                    "error": _join_errors(
                        base.error, f"judge unavailable: {type(exc).__name__}: {exc}"
                    ),
                    "duration_ms": base.duration_ms + (time.perf_counter() - started) * 1000.0,
                }
            )

        verdict = (result.text or "").strip()
        unsafe = verdict.lower().startswith("unsafe")
        duration_ms = base.duration_ms + (time.perf_counter() - started) * 1000.0

        if not unsafe:
            # Judge agrees the text is clean. Rule findings still stand.
            return base.model_copy(
                update={
                    "judge_used": True,
                    "judge_verdict": verdict[:200],
                    "duration_ms": duration_ms,
                    "explanation": f"{base.explanation} Judge: agrees no violation.",
                }
            )

        judged = PolicyViolation(
            policy_id="JUDGE-01",
            category="llm_judge",
            severity=Severity.HIGH,
            reason=f"LLM judge flagged the response: {verdict[:160]}",
            matched_preview="",
        )
        violations = [*base.violations, judged]
        peak = max((v.severity for v in violations), key=severity_rank)
        return base.model_copy(
            update={
                "status": SignalStatus.FAIL,
                "score": max(base.score, _SEVERITY_SCORE[peak]),
                "severity": peak,
                "violations": violations,
                "judge_used": True,
                "judge_verdict": verdict[:200],
                "duration_ms": duration_ms,
                "explanation": f"{base.explanation} Judge flagged the response.",
            }
        )


def _join_errors(existing: str | None, new: str) -> str:
    return f"{existing}; {new}" if existing else new


__all__ = [
    "SafetyRule",
    "SafetyService",
    "cached_safety_rules",
    "load_safety_rules",
]
