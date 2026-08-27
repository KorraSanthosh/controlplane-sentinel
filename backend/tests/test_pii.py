"""PII detection and redaction.

Precision matters more than recall here, because a false positive corrupts a correct answer
mid-sentence. The negative cases below are therefore as load-bearing as the positive ones — an
order number must not become ``[REDACTED:ACCOUNT_NUMBER]``.

Recall is knowingly bounded: regex catches structured identifiers and misses free-form ones.
``test_unstructured_name_is_not_caught`` records that limit as a fact about the system rather
than leaving it to be discovered in a demo.
"""

from __future__ import annotations

import pytest

from app.schemas.signals import PIIKind, Severity, SignalStatus
from app.services.pii.service import REDACTION_TEMPLATE, PIIService, mask_value


@pytest.fixture
def pii() -> PIIService:
    return PIIService()


# ---------------------------------------------------------------------------
# Detection — positive cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Reach them at j.doe@northwind.example.com for details.", PIIKind.EMAIL),
        ("Call back on (415) 555-0142 tomorrow.", PIIKind.PHONE),
        ("Call back on +1 415-555-0142 tomorrow.", PIIKind.PHONE),
        ("The account number is NW-4417-0092.", PIIKind.ACCOUNT_NUMBER),
        ("Card on file 4111 1111 1111 1111 expires soon.", PIIKind.CREDIT_CARD),
        ("SSN 123-45-6789 on record.", PIIKind.NATIONAL_ID),
        ("Last seen from 203.0.113.42 this morning.", PIIKind.IP_ADDRESS),
        ("Ship to 1200 Market Street, Springfield.", PIIKind.POSTAL_ADDRESS),
        ("Date of birth: 1987-03-14 per the file.", PIIKind.DATE_OF_BIRTH),
    ],
)
def test_detects_structured_identifiers(pii: PIIService, text: str, kind: PIIKind) -> None:
    signal = pii.scan(text)
    assert signal.detected
    assert kind.value in signal.counts, f"{kind.value} not in {signal.counts}"


def test_clean_text_passes(pii: PIIService) -> None:
    signal = pii.scan("The Plus plan includes 40 GB of high-speed data per month.")
    assert signal.status is SignalStatus.PASS
    assert not signal.detected
    assert signal.score == 0.0
    assert signal.severity is Severity.NONE


def test_empty_text_is_handled(pii: PIIService) -> None:
    assert not pii.scan("").detected


# ---------------------------------------------------------------------------
# Detection — precision
# ---------------------------------------------------------------------------
def test_invalid_card_number_is_rejected_by_luhn(pii: PIIService) -> None:
    signal = pii.scan("Reference 1234 5678 9012 3456 for the ticket.")
    assert "credit_card" not in signal.counts


def test_bare_date_is_not_personal_data(pii: PIIService) -> None:
    """A date needs a context cue before it is a date of birth."""
    signal = pii.scan("The outage started on 2024-03-14 and was resolved the same day.")
    assert "date_of_birth" not in signal.counts


def test_plan_price_is_not_an_identifier(pii: PIIService) -> None:
    signal = pii.scan("The Premium plan costs $65 per month and includes 100 GB.")
    assert not signal.detected, f"false positives: {signal.counts}"


def test_version_string_is_not_an_ip_address(pii: PIIService) -> None:
    signal = pii.scan("Firmware 10.2.1 shipped last week.")
    assert "ip_address" not in signal.counts


def test_unstructured_name_is_not_caught(pii: PIIService) -> None:
    """A documented limit, asserted so it cannot be mistaken for working coverage."""
    signal = pii.scan("The customer's name is John Doe.")
    assert not signal.detected


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------
def test_card_number_is_not_partly_claimed_by_the_phone_detector(pii: PIIService) -> None:
    signal = pii.scan("Card on file 4111 1111 1111 1111.")
    assert signal.counts == {"credit_card": 1}
    assert len(signal.matches) == 1


def test_matches_are_ordered_by_position(pii: PIIService) -> None:
    signal = pii.scan("Email a@b.co, phone (415) 555-0142, account NW-4417-0092.")
    starts = [m.start for m in signal.matches]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_severity_follows_the_most_sensitive_kind(pii: PIIService) -> None:
    assert pii.scan("Last seen from 203.0.113.42.").severity is Severity.LOW
    assert pii.scan("Card 4111 1111 1111 1111.").severity is Severity.CRITICAL


def test_a_card_outranks_several_emails(pii: PIIService) -> None:
    emails = pii.scan("a@x.com and b@x.com and c@x.com are on the thread.")
    card = pii.scan("Card on file 4111 1111 1111 1111.")
    assert card.score > emails.score


def test_breadth_raises_the_score_but_does_not_dominate(pii: PIIService) -> None:
    one = pii.scan("Email a@x.com.")
    many = pii.scan("Email a@x.com, phone (415) 555-0142, from 203.0.113.42.")
    assert many.score > one.score
    assert many.score <= 1.0


def test_score_never_exceeds_one(pii: PIIService) -> None:
    signal = pii.scan(
        "Card 4111 1111 1111 1111, SSN 123-45-6789, email a@x.com, "
        "phone (415) 555-0142, account NW-4417-0092, from 203.0.113.42, "
        "at 1200 Market Street, date of birth: 1987-03-14."
    )
    assert signal.score <= 1.0
    assert signal.severity is Severity.CRITICAL


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------
def test_redaction_masks_every_span(pii: PIIService) -> None:
    text = "Write to j.doe@northwind.example.com or call (415) 555-0142."
    signal = pii.scan(text)
    redacted = pii.redact(text, signal.matches)
    assert "j.doe@northwind.example.com" not in redacted
    assert "555-0142" not in redacted
    assert REDACTION_TEMPLATE.format(kind="EMAIL") in redacted
    assert REDACTION_TEMPLATE.format(kind="PHONE") in redacted


def test_redaction_preserves_surrounding_text(pii: PIIService) -> None:
    text = "Your refund of $42 goes to j.doe@x.com within 5 days."
    redacted = pii.redact(text, pii.scan(text).matches)
    assert redacted.startswith("Your refund of $42 goes to ")
    assert redacted.endswith(" within 5 days.")


def test_redacted_text_is_clean_on_rescan(pii: PIIService) -> None:
    """The strongest available check: nothing detectable survives a round trip."""
    text = (
        "Account NW-4417-0092, email j.doe@x.com, phone (415) 555-0142, "
        "card 4111 1111 1111 1111, SSN 123-45-6789, date of birth: 1987-03-14."
    )
    redacted = pii.redact(text, pii.scan(text).matches)
    assert not pii.scan(redacted).detected, f"survived: {pii.scan(redacted).counts}"


def test_redaction_of_nothing_returns_the_input(pii: PIIService) -> None:
    text = "Nothing sensitive here."
    assert pii.redact(text, []) == text


def test_offsets_stay_valid_across_multiple_replacements(pii: PIIService) -> None:
    """Right-to-left application is what keeps earlier offsets correct."""
    text = "a@x.com then b@y.com then c@z.com"
    redacted = pii.redact(text, pii.scan(text).matches)
    assert redacted.count(REDACTION_TEMPLATE.format(kind="EMAIL")) == 3
    assert "@x.com" not in redacted and "@z.com" not in redacted


# ---------------------------------------------------------------------------
# Storage safety (NFR-01)
# ---------------------------------------------------------------------------
def test_match_previews_never_carry_the_raw_value(pii: PIIService) -> None:
    text = "Card on file 4111 1111 1111 1111, email j.doe@northwind.example.com."
    signal = pii.scan(text)
    for match in signal.matches:
        assert "4111 1111 1111 1111" not in match.preview
        assert "j.doe@northwind" not in match.preview


def test_mask_value_keeps_only_the_last_four_digits() -> None:
    assert mask_value(PIIKind.CREDIT_CARD, "4111 1111 1111 1111").endswith("1111")
    assert "4111 1111" not in mask_value(PIIKind.CREDIT_CARD, "4111 1111 1111 1111")


def test_mask_value_keeps_the_email_domain_only() -> None:
    masked = mask_value(PIIKind.EMAIL, "john.doe@northwind.example.com")
    assert masked.endswith("@northwind.example.com")
    assert "john.doe" not in masked


def test_mask_for_storage_truncates_and_says_so(pii: PIIService) -> None:
    long_text = "Contact j.doe@x.com. " + ("filler " * 200)
    stored = pii.mask_for_storage(long_text, limit=100)
    assert "j.doe@x.com" not in stored
    assert "truncated" in stored
    assert len(stored) < len(long_text)


def test_mask_for_storage_masks_before_truncating(pii: PIIService) -> None:
    """Truncating first could cut a span in half and leave a partial identifier behind."""
    text = "x" * 90 + " card 4111 1111 1111 1111 tail"
    stored = pii.mask_for_storage(text, limit=200)
    assert "4111" not in stored
