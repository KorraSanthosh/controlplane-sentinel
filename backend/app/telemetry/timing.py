"""Stage timing helpers.

The brief asks how a checker protects latency. The honest answer requires measurement, so
every pipeline stage is timed and the control-plane overhead is reported separately from the
model's own latency.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from app.schemas.signals import StageTiming


class StageTimer:
    """Accumulates named stage durations for one request."""

    def __init__(self) -> None:
        self._stages: list[StageTiming] = []
        self._started = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._stages.append(
                StageTiming(stage=name, duration_ms=(time.perf_counter() - start) * 1000.0)
            )

    def record(self, name: str, duration_ms: float) -> None:
        self._stages.append(StageTiming(stage=name, duration_ms=duration_ms))

    @property
    def stages(self) -> list[StageTiming]:
        return list(self._stages)

    def total_ms(self) -> float:
        """Wall-clock since the timer was created."""
        return (time.perf_counter() - self._started) * 1000.0

    def sum_except(self, *excluded: str) -> float:
        """Sum of recorded stages, skipping the named ones.

        Used to derive control-plane overhead by excluding the model call — the model's
        latency is not ours to claim credit or blame for.
        """
        return sum(s.duration_ms for s in self._stages if s.stage not in excluded)


__all__ = ["StageTimer"]
