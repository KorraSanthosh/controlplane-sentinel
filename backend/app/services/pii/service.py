"""PII detection and redaction.

Deliberately rule-based (regex + a Luhn check) rather than model-based. Three reasons:

* it is deterministic, so the demo and tests are reproducible (NFR-06);
* it is fast enough to run inline on every request — this is the fast path;
* it needs no network, so it cannot fail open when a provider is down.

The honest limitation is recall: regex catches structured identifiers well and unstructured
ones (free-form names, informal addresses) poorly. The upgrade path is a NER model such as
Presidio or spaCy on the deep path; that is documented in the README rather than implied to
already work here.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from app.schemas.signals import (
    PIIKind,
    PIIMatch,
    PIISignal,
    Severity,
    SignalStatus,
    severity_rank,
)

REDACTION_TEMPLATE = "[REDACTED:{kind}]"


@dataclass(frozen=True)
class Detector:
    kind: PIIKind
    pattern: re.Pattern[str]
    #: Which regex group holds the sensitive span. 0 = whole match. Used where a context
    #: cue must be present but should not itself be redacted (e.g. "date of birth: X").
    group: int
    severity: Severity
    #: Per-kind risk contribution, 0..1.
    weight: float
    #: Higher wins when two detectors claim overlapping spans.
    priority: int
    confidence: float = 1.0
    validator: str | None = None


_DETECTORS: tuple[Detector, ...] = (
    Detector(
        kind=PIIKind.EMAIL,
        pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        group=0,
        severity=Severity.MEDIUM,
        weight=0.55,
        priority=90,
    ),
    Detector(
        kind=PIIKind.CREDIT_CARD,
        # Broad digit-group match, then Luhn-validated to keep precision high.
        pattern=re.compile(r"\b\d(?:[ -]?\d){12,18}\b"),
        group=0,
        severity=Severity.CRITICAL,
        weight=1.0,
        priority=80,
        validator="luhn",
    ),
    Detector(
        kind=PIIKind.NATIONAL_ID,
        pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        group=0,
        severity=Severity.CRITICAL,
        weight=0.95,
        priority=70,
    ),
    Detector(
        kind=PIIKind.ACCOUNT_NUMBER,
        # Northwind's documented account format, plus a generic contextual form.
        pattern=re.compile(
            r"\bNW-\d{4}-\d{4}\b"
            r"|(?:account(?:\s+number)?|acct\.?)\s*[:#]?\s*([A-Z]{0,3}-?\d[\d-]{5,})",
            re.IGNORECASE,
        ),
        group=0,
        severity=Severity.HIGH,
        weight=0.75,
        priority=60,
    ),
    Detector(
        kind=PIIKind.PHONE,
        pattern=re.compile(
            r"(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}\b"
        ),
        group=0,
        severity=Severity.MEDIUM,
        weight=0.55,
        priority=50,
    ),
    Detector(
        kind=PIIKind.DATE_OF_BIRTH,
        # Requires a context cue: a bare date is not inherently personal data.
        pattern=re.compile(
            r"(?:date\s+of\s+birth|d\.?o\.?b\.?|born(?:\s+on)?)\s*[:is]{0,3}\s*"
            r"((?:19|20)\d{2}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            re.IGNORECASE,
        ),
        group=1,
        severity=Severity.HIGH,
        weight=0.70,
        priority=40,
    ),
    Detector(
        kind=PIIKind.IP_ADDRESS,
        pattern=re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
        ),
        group=0,
        severity=Severity.LOW,
        weight=0.30,
        priority=30,
    ),
    Detector(
        kind=PIIKind.POSTAL_ADDRESS,
        pattern=re.compile(
            r"\b\d{1,6}\s+(?:[A-Z][A-Za-z.]*\s+){1,4}"
            r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Court|Ct|Plaza|Terrace)\b"
            r"(?:,\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)?"
            r"(?:,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?"
        ),
        group=0,
        severity=Severity.MEDIUM,
        weight=0.60,
        priority=20,
        confidence=0.8,
    ),
)


def _luhn_valid(raw: str) -> bool:
    digits = [int(c) for c in raw if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_VALIDATORS = {"luhn": _luhn_valid}


def mask_value(kind: PIIKind, value: str) -> str:
    """Produce a masked preview. The raw value never leaves this function."""
    if kind is PIIKind.EMAIL and "@" in value:
        local, _, domain = value.partition("@")
        head = local[0] if local else "*"
        return f"{head}{'*' * max(len(local) - 1, 1)}@{domain}"
    digits_only = [c for c in value if c.isdigit()]
    if kind in (PIIKind.CREDIT_CARD, PIIKind.PHONE, PIIKind.ACCOUNT_NUMBER) and len(digits_only) >= 4:
        return f"{'*' * (len(digits_only) - 4)}{''.join(digits_only[-4:])}"
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[0]}{'*' * (len(value) - 2)}{value[-1]}"


@dataclass
class _RawMatch:
    kind: PIIKind
    start: int
    end: int
    value: str
    severity: Severity
    weight: float
    priority: int
    confidence: float


def _collect(text: str) -> list[_RawMatch]:
    found: list[_RawMatch] = []
    for det in _DETECTORS:
        for m in det.pattern.finditer(text):
            group = det.group
            # A group may be optional in an alternation; fall back to the whole match.
            if group and m.group(group) is None:
                group = 0
            start, end = m.span(group)
            value = m.group(group)
            if not value:
                continue
            if det.validator and not _VALIDATORS[det.validator](value):
                continue
            found.append(
                _RawMatch(
                    kind=det.kind,
                    start=start,
                    end=end,
                    value=value,
                    severity=det.severity,
                    weight=det.weight,
                    priority=det.priority,
                    confidence=det.confidence,
                )
            )
    return found


def _resolve_overlaps(matches: list[_RawMatch]) -> list[_RawMatch]:
    """Keep the strongest claim on any overlapping span.

    Ordering is longest-span first, then detector priority. A 16-digit card number would
    otherwise be partly claimed by the phone detector.
    """
    ordered = sorted(matches, key=lambda m: (-(m.end - m.start), -m.priority, m.start))
    kept: list[_RawMatch] = []
    for cand in ordered:
        if any(cand.start < k.end and k.start < cand.end for k in kept):
            continue
        kept.append(cand)
    return sorted(kept, key=lambda m: m.start)


class PIIService:
    """Detects and redacts personal data in model output."""

    name = "pii"

    def scan(self, text: str) -> PIISignal:
        started = time.perf_counter()
        raw = _resolve_overlaps(_collect(text or ""))

        matches = [
            PIIMatch(
                kind=r.kind,
                start=r.start,
                end=r.end,
                preview=mask_value(r.kind, r.value),
                confidence=r.confidence,
            )
            for r in raw
        ]
        counts: dict[str, int] = {}
        for r in raw:
            counts[r.kind.value] = counts.get(r.kind.value, 0) + 1

        duration_ms = (time.perf_counter() - started) * 1000.0

        if not raw:
            return PIISignal(
                status=SignalStatus.PASS,
                score=0.0,
                severity=Severity.NONE,
                detected=False,
                redactable=False,
                explanation="No PII patterns detected.",
                duration_ms=duration_ms,
            )

        # Risk is driven by the most sensitive kind found, nudged up by breadth. A card
        # number alone outranks three email addresses.
        peak = max(r.weight for r in raw)
        breadth_bonus = min(0.10, 0.02 * (len(counts) - 1))
        score = min(1.0, peak + breadth_bonus)
        severity = max((r.severity for r in raw), key=severity_rank)

        kinds = ", ".join(sorted(counts))
        return PIISignal(
            status=SignalStatus.FAIL,
            score=score,
            severity=severity,
            detected=True,
            redactable=True,  # every regex match carries offsets, so all spans are maskable
            matches=matches,
            counts=counts,
            explanation=f"Detected {len(raw)} PII span(s) across {len(counts)} categor(y/ies): {kinds}.",
            duration_ms=duration_ms,
        )

    def redact(self, text: str, matches: list[PIIMatch]) -> str:
        """Replace each detected span with a typed placeholder.

        Applied right-to-left so earlier offsets stay valid as the string changes length.
        """
        if not matches:
            return text
        out = text
        for m in sorted(matches, key=lambda x: x.start, reverse=True):
            out = out[: m.start] + REDACTION_TEMPLATE.format(kind=m.kind.value.upper()) + out[m.end :]
        return out

    def mask_for_storage(self, text: str, limit: int = 400) -> str:
        """Mask any PII in free text before it is persisted or logged (NFR-01).

        Used by the audit service and the logging filter, so a record built from a
        PII-bearing response cannot carry the original values.
        """
        try:
            signal = self.scan(text or "")
            masked = self.redact(text or "", signal.matches)
        except Exception:
            return "[masking failed - preview suppressed]"
        if len(masked) > limit:
            return masked[:limit] + f"... [truncated, {len(masked)} chars]"
        return masked


__all__ = ["PIIService", "REDACTION_TEMPLATE", "mask_value"]
