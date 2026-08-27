"""Unit tests for the Bias and Fairness evaluation service."""

from __future__ import annotations

import pytest

from app.schemas.signals import Severity, SignalStatus
from app.services.bias.service import BiasRule, BiasService


def test_bias_clean_response() -> None:
    service = BiasService()
    text = "The Northwind Plus plan costs $45 per month and includes 40 GB of high-speed data."
    signal = service.check(text)

    assert signal.status == SignalStatus.PASS
    assert signal.score == 0.0
    assert signal.severity == Severity.NONE
    assert len(signal.findings) == 0
    assert signal.groups_implicated == []


def test_bias_gender_detection() -> None:
    service = BiasService()
    text = "Please note that female applicants require spouse approval for account verification."
    signal = service.check(text)

    assert signal.status == SignalStatus.FAIL
    assert signal.score >= 0.75
    assert signal.severity == Severity.HIGH
    assert len(signal.findings) == 1
    assert signal.findings[0].policy_id == "BIAS-GENDER-01"
    assert signal.findings[0].category == "gender_bias"
    assert "gender" in signal.groups_implicated


def test_bias_age_detection() -> None:
    service = BiasService()
    text = "We cannot process this upgrade as senior citizens are higher risk for digital plans."
    signal = service.check(text)

    assert signal.status == SignalStatus.FAIL
    assert signal.severity == Severity.HIGH
    assert len(signal.findings) == 1
    assert signal.findings[0].policy_id == "BIAS-AGE-01"
    assert "age" in signal.groups_implicated


def test_bias_location_proxy_detection() -> None:
    service = BiasService()
    text = "Your application was declined because your postcode blacklisted due to demographic risk factors."
    signal = service.check(text)

    assert signal.status == SignalStatus.FAIL
    assert signal.severity == Severity.HIGH
    assert len(signal.findings) == 1
    assert signal.findings[0].category == "income_location_proxy"
    assert "postcode_proxy" in signal.groups_implicated


def test_bias_multiple_violations() -> None:
    service = BiasService()
    text = (
        "Women are not eligible for this tier, and applicants from low-income neighborhood restriction "
        "are excluded automatically."
    )
    signal = service.check(text)

    assert signal.status == SignalStatus.FAIL
    assert len(signal.findings) == 2
    assert sorted(signal.groups_implicated) == ["gender", "postcode_proxy"]


def test_bias_enabled_categories_filter() -> None:
    service = BiasService()
    text = "Women are not eligible for this promotion."
    # Filter to only run age_bias
    signal = service.check(text, enabled_categories={"age_bias"})

    assert signal.status == SignalStatus.PASS
    assert len(signal.findings) == 0
