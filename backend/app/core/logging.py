"""Logging setup, including a filter that masks PII in log output.

Two things go wrong with logging in a system like this, and both are worth designing against.

**Logs are a PII leak.** Detectors run on text that has just been found to contain personal
data. Any log line that quotes the response — an exception traceback, a debug dump — is a copy
of that data in a file nobody redacts later. :class:`PIIRedactingFilter` runs the same detector
used for the audit previews across every formatted record, so a leak has to get past the same
check the stored payloads do.

**Logs are the only evidence of a governance gap.** When the audit store is down, the ERROR line
is the record. So the filter must never raise: a failure inside redaction falls back to a fixed
placeholder rather than dropping the line or crashing the handler.

The honest limitation is inherited from the detector: regex catches structured identifiers, not
free-form ones. This reduces exposure; it does not make logs safe to ship to a third party.
"""

from __future__ import annotations

import logging
import logging.config
from typing import Any

from app.services.pii.service import PIIService

#: Substituted when redaction itself fails. Losing the message text is acceptable; leaking it
#: or dropping the line is not.
UNREDACTABLE = "[log message withheld: redaction failed]"

#: Loggers that are noisy at INFO and say nothing this project needs.
_QUIET = ("httpx", "httpcore", "neo4j", "pymongo", "urllib3", "anthropic")


class PIIRedactingFilter(logging.Filter):
    """Masks PII in the final formatted message of every record.

    Rewrites ``record.msg`` to the already-interpolated, masked text and clears ``args`` —
    otherwise a ``%s`` in the masked text would be re-interpolated against arguments that are
    themselves unmasked.
    """

    def __init__(self, pii: PIIService | None = None, limit: int = 2000) -> None:
        super().__init__()
        self.pii = pii or PIIService()
        self.limit = limit

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            masked = self.pii.mask_for_storage(message, limit=self.limit)
            if masked != message:
                record.msg = masked
                record.args = ()
        except Exception:  # noqa: BLE001 - a logging filter must never break logging
            record.msg = UNREDACTABLE
            record.args = ()
        return True


def logging_config(level: str = "INFO") -> dict[str, Any]:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"pii": {"()": PIIRedactingFilter}},
        "formatters": {
            "standard": {
                "format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                "datefmt": "%H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "filters": ["pii"],
                "stream": "ext://sys.stdout",
            }
        },
        "root": {"level": level.upper(), "handlers": ["console"]},
        "loggers": {name: {"level": "WARNING"} for name in _QUIET},
    }


def configure_logging(level: str = "INFO") -> None:
    logging.config.dictConfig(logging_config(level))


__all__ = ["UNREDACTABLE", "PIIRedactingFilter", "configure_logging", "logging_config"]
