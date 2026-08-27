"""``/metrics`` — governance and latency aggregates for the dashboard.

Two numbers here are the honest answer to the brief's latency question:
``p95_controlplane_overhead_ms`` (what governance costs) and ``deep_path_rate`` (how often the
expensive path was actually needed). Both are computed from real measurements on real requests,
not asserted.

Aggregation is over the most recent ``METRICS_WINDOW`` records, in the API process. A
prototype-scale simplification — correct for the demo, and the point where a real deployment
would move to a server-side aggregation pipeline.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import AuditDep
from app.schemas.audit import MetricsSummary
from app.schemas.decision import Decision
from app.services.audit.repository import AuditFilter
from app.services.audit.service import METRICS_WINDOW

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    response_model=MetricsSummary,
    summary=f"Decision, risk and latency aggregates over the last {METRICS_WINDOW} records",
)
async def metrics(
    audit: AuditDep,
    use_case: str | None = None,
    decision: Decision | None = None,
    min_risk: float | None = Query(default=None, ge=0.0, le=1.0),
) -> MetricsSummary:
    return await audit.metrics(
        filter_=AuditFilter(decision=decision, use_case=use_case, min_risk=min_risk)
    )


__all__ = ["router"]
