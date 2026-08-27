"""Claim extraction.

Turns a free-text model response into ``(subject, predicate, value)`` triples that can be
checked against the graph.

The approach is rule-based: locate known entity mentions, then pull predicate values out of
the same sentence, assigning each value to the *nearest* entity mention. Proximity matters —
"Basic and Plus carry a 14-day refund window, while Premium carries a 30-day refund window"
contains three subjects and two values in one sentence, and a naive extractor would attach
the wrong number to Premium and report a contradiction that isn't there.

Known limits, stated plainly rather than papered over:

* recall is bounded by the pattern list — an assertion phrased unusually is not extracted,
  and therefore not checked. This is the system's main false-negative channel.
* only predicates present in the graph are verifiable; anything else is counted as
  unverifiable rather than judged.

The upgrade path is LLM-based claim extraction on the deep path, which would raise recall at
the cost of determinism and latency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.grounding.graph_repo import GraphEntity

#: How far back from a boolean cue to look for a negation.
_NEGATION_WINDOW = 40

_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|excludes?|excluded|requires?|"
    r"doesn't|does\s+not|isn't|is\s+not|aren't|are\s+not|don't|do\s+not)\b",
    re.IGNORECASE,
)

#: Capitalised product-ish phrase followed by a category noun. Used to notice a subject the
#: graph has never heard of — the difference between "we can refute this" and "we have
#: nothing on this". Group 1 is the phrase, group 2 the category noun.
_CANDIDATE_ENTITY_RE = re.compile(
    r"\b((?:[A-Z][a-zA-Z]+)(?:\s+[A-Z][a-zA-Z]+){0,3})"
    r"\s+(tier|plan|package|bundle|add-on|addon)\b"
)

#: Leading words that are not part of a product name. A sentence-initial "The" is capitalised
#: like a brand, so "The Plus plan" captures as "The Plus" — and comparing that against the
#: alias table would report a plan the graph knows perfectly well as an unknown subject, which
#: turns a grounded answer into a spurious FLAG. Stripped for matching *and* for display.
_LEADING_NOISE = frozenset(
    {"the", "a", "an", "our", "your", "their", "this", "that", "these", "those", "new",
     "current", "standard"}
)

#: Optional organisation prefix. Prose alternates between "the Northwind Plus plan" and "the Plus
#: plan" for the same product, so both spellings are tried against the alias table. This is
#: demo-domain vocabulary; a production extractor would read it from configuration alongside the
#: graph, and the display name keeps the brand either way.
_BRAND_PREFIX_RE = re.compile(r"^northwind\s+", re.IGNORECASE)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _phrase_variants(phrase: str, noun: str) -> tuple[set[str], str]:
    """Candidate surface forms for a captured phrase, plus the form worth showing a human.

    "The Northwind Plus" + "plan" yields ``{the northwind plus plan, the northwind plus,
    northwind plus plan, northwind plus, plus plan, plus}`` — every alias spelling the seed data
    plausibly uses. The graph is only asked about a subject when *none* of these are known;
    matching on a subset would invent unknown subjects out of ordinary English.

    The returned display form keeps the brand but drops leading determiners, so an unknown
    subject is reported as "Enterprise Fiber" rather than "The Enterprise Fiber".
    """
    tokens = phrase.split()
    variants: set[str] = set()
    display = phrase
    stripped_any = False
    while tokens:
        stem = " ".join(tokens)
        variants.add(stem.lower())
        variants.add(f"{stem} {noun}".lower())
        if stripped_any:
            display = stem
        if tokens[0].lower() not in _LEADING_NOISE:
            break
        tokens = tokens[1:]
        stripped_any = True
    # The brand name is optional in prose ("the Northwind Plus plan" / "the Plus plan") and the
    # alias table may spell it either way, so try both against every variant.
    variants |= {_BRAND_PREFIX_RE.sub("", v) for v in variants}
    return {v for v in variants if v}, display


@dataclass(frozen=True)
class PredicateSpec:
    """How to recognise one predicate's value in prose."""

    predicate: str
    value_type: str  # "number" | "boolean"
    label: str
    #: Numeric predicates: group 1 holds the value.
    #: Boolean predicates: the whole match is the cue; polarity comes from negation lookback.
    patterns: tuple[re.Pattern[str], ...]


PREDICATE_SPECS: tuple[PredicateSpec, ...] = (
    PredicateSpec(
        predicate="MONTHLY_PRICE_USD",
        value_type="number",
        label="monthly price",
        patterns=(
            re.compile(
                r"\$\s?(\d+(?:\.\d+)?)\s*(?:per\s+month|/\s?month|a\s+month|monthly)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:costs?|priced\s+at|is)\s+\$\s?(\d+(?:\.\d+)?)\b(?![\s\S]{0,12}?one[-\s]time)",
                re.IGNORECASE,
            ),
        ),
    ),
    PredicateSpec(
        predicate="INCLUDED_DATA_GB",
        value_type="number",
        label="included data allowance",
        patterns=(re.compile(r"(\d+(?:\.\d+)?)\s*GB\b", re.IGNORECASE),),
    ),
    PredicateSpec(
        predicate="REFUND_WINDOW_DAYS",
        value_type="number",
        label="refund window",
        patterns=(
            re.compile(r"(\d+)[\s-]*day\s+refund", re.IGNORECASE),
            re.compile(r"refund\s+window[^.]{0,30}?(\d+)\s*days?", re.IGNORECASE),
        ),
    ),
    PredicateSpec(
        predicate="UPGRADE_ELIGIBILITY_MONTHS",
        value_type="number",
        label="upgrade eligibility period",
        patterns=(
            re.compile(r"after\s+(\d+)\s+months?", re.IGNORECASE),
            re.compile(r"(\d+)\s+months?\s+of\s+active\s+service", re.IGNORECASE),
        ),
    ),
    PredicateSpec(
        predicate="INCLUDES_INTERNATIONAL_ROAMING",
        value_type="boolean",
        label="international roaming inclusion",
        patterns=(re.compile(r"international\s+roaming", re.IGNORECASE),),
    ),
)


@dataclass
class ExtractedClaim:
    """One assertion pulled out of the response."""

    sentence: str
    subject_id: str | None  # None => entity not present in the graph
    subject_name: str
    predicate: str | None  # None => "this entity exists / has properties" style claim
    value: str | None
    value_type: str
    label: str = ""

    def key(self) -> tuple:
        return (self.subject_id, self.subject_name.lower(), self.predicate, self.value)


@dataclass
class ExtractionResult:
    claims: list[ExtractedClaim] = field(default_factory=list)
    mentioned_entity_ids: list[str] = field(default_factory=list)
    sentences_scanned: int = 0


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text or "") if s.strip()]


def _boolean_polarity(sentence: str, cue_start: int) -> bool:
    """True when the sentence asserts the cue positively.

    Only text *before* the cue counts. "includes unlimited international roaming at no extra
    cost" is a positive claim — the "no" belongs to the cost clause that follows, and reading
    it as a negation would flip a hallucination into an apparent match.
    """
    window = sentence[max(0, cue_start - _NEGATION_WINDOW) : cue_start]
    return _NEGATION_RE.search(window) is None


def _alias_index(entities: list[GraphEntity]) -> list[tuple[str, GraphEntity, re.Pattern[str]]]:
    """Longest aliases first, so "plus plan" wins over "plus"."""
    index: list[tuple[str, GraphEntity, re.Pattern[str]]] = []
    for ent in entities:
        for alias in sorted(ent.alias_set(), key=len, reverse=True):
            index.append(
                (alias, ent, re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE))
            )
    index.sort(key=lambda t: len(t[0]), reverse=True)
    return index


def _known_alias_set(entities: list[GraphEntity]) -> set[str]:
    known: set[str] = set()
    for ent in entities:
        known |= ent.alias_set()
        known.add(ent.id.lower())
    return known


def extract_claims(text: str, entities: list[GraphEntity]) -> ExtractionResult:
    """Extract verifiable triples plus mentions of unknown subjects."""
    result = ExtractionResult()
    if not text:
        return result

    alias_index = _alias_index(entities)
    known_aliases = _known_alias_set(entities)
    sentences = split_sentences(text)
    result.sentences_scanned = len(sentences)

    seen: set[tuple] = set()
    mentioned: list[str] = []

    for sentence in sentences:
        # --- which known entities does this sentence talk about? ---
        mentions: dict[str, tuple[GraphEntity, int]] = {}
        claimed_spans: list[tuple[int, int]] = []
        for alias, ent, pattern in alias_index:
            m = pattern.search(sentence)
            if not m:
                continue
            # A shorter alias inside an already-claimed longer alias adds nothing.
            if any(m.start() >= s and m.end() <= e for s, e in claimed_spans):
                continue
            claimed_spans.append(m.span())
            if ent.id not in mentions:
                mentions[ent.id] = (ent, m.start())
                if ent.id not in mentioned:
                    mentioned.append(ent.id)

        # --- predicate values present in this sentence ---
        found: dict[str, list[tuple[str, int]]] = {}
        for spec in PREDICATE_SPECS:
            hits: list[tuple[str, int]] = []
            for pattern in spec.patterns:
                for m in pattern.finditer(sentence):
                    if spec.value_type == "boolean":
                        value = "true" if _boolean_polarity(sentence, m.start()) else "false"
                    else:
                        value = m.group(1)
                    hits.append((value, m.start()))
            if hits:
                found[spec.predicate] = hits

        # For booleans, any negated mention makes the whole sentence's claim negative:
        # the model is stating a restriction, and assuming the positive reading would
        # accuse it of a claim it did not make.
        for spec in PREDICATE_SPECS:
            if spec.value_type == "boolean" and spec.predicate in found:
                hits = found[spec.predicate]
                if any(v == "false" for v, _ in hits):
                    found[spec.predicate] = [("false", pos) for _, pos in hits]

        # --- pair each value with its nearest subject ---
        for ent, ent_pos in mentions.values():
            for spec in PREDICATE_SPECS:
                hits = found.get(spec.predicate)
                if not hits:
                    continue
                value, _pos = min(hits, key=lambda h: abs(h[1] - ent_pos))
                claim = ExtractedClaim(
                    sentence=sentence,
                    subject_id=ent.id,
                    subject_name=ent.name,
                    predicate=spec.predicate,
                    value=value,
                    value_type=spec.value_type,
                    label=spec.label,
                )
                if claim.key() in seen:
                    continue
                seen.add(claim.key())
                result.claims.append(claim)

        # --- subjects the graph has never heard of ---
        for m in _CANDIDATE_ENTITY_RE.finditer(sentence):
            variants, display = _phrase_variants(m.group(1).strip(), m.group(2))
            if variants & known_aliases:
                continue
            claim = ExtractedClaim(
                sentence=sentence,
                subject_id=None,
                subject_name=display,
                predicate=None,
                value=None,
                value_type="entity",
                label="unknown subject",
            )
            if claim.key() in seen:
                continue
            seen.add(claim.key())
            result.claims.append(claim)

    result.mentioned_entity_ids = mentioned
    return result


__all__ = [
    "PREDICATE_SPECS",
    "ExtractedClaim",
    "ExtractionResult",
    "PredicateSpec",
    "extract_claims",
    "split_sentences",
]
