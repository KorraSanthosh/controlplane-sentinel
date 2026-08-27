"""``/audits`` — the decision trail (FR-10).

Reads only. Records are immutable: a decision is evidence of what the system did at a point in
time under a named policy version, so nothing here edits one. Reviewer disagreement is captured
as a *separate* feedback record pointing at the original, which keeps both the original judgement
and the correction.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.deps import AuditDep
from app.schemas.audit import AuditRecord, AuditSummary, FeedbackRecord
from app.schemas.decision import Decision
from app.services.audit.repository import AuditFilter

router = APIRouter(tags=["audits"])


class AuditPage(BaseModel):
    items: list[AuditSummary]
    total: int
    limit: int
    offset: int


class FeedbackRequest(BaseModel):
    """A human reviewer's verdict on a decision the system made."""

    reviewer: str = Field(min_length=1, max_length=120)
    override_decision: Decision | None = None
    agrees_with_decision: bool | None = None
    comment: str = Field(default="", max_length=2000)


@router.get("/audits", response_model=AuditPage, summary="List decision records, newest first")
async def list_audits(
    audit: AuditDep,
    decision: Decision | None = None,
    use_case: str | None = None,
    min_risk: float | None = Query(default=None, ge=0.0, le=1.0),
    requires_human_review: bool | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> AuditPage:
    filter_ = AuditFilter(
        decision=decision,
        use_case=use_case,
        min_risk=min_risk,
        requires_human_review=requires_human_review,
    )
    items, total = await audit.list_summaries(filter_=filter_, limit=limit, offset=offset)
    return AuditPage(items=items, total=total, limit=limit, offset=offset)


@router.get(
    "/audits/{request_id}",
    response_model=AuditRecord,
    summary="One full decision record, including every signal and fired rule",
)
async def get_audit(request_id: str, audit: AuditDep) -> AuditRecord:
    record = await audit.get(request_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No audit record for request '{request_id}'. Records are written "
                f"asynchronously, so a very recent request may not have landed yet — or the "
                f"audit store was unavailable when it was made (see /health)."
            ),
        )
    return record


@router.get(
    "/audits/{request_id}/feedback",
    response_model=list[FeedbackRecord],
    summary="Reviewer feedback attached to a decision",
)
async def get_feedback(request_id: str, audit: AuditDep) -> list[FeedbackRecord]:
    return await audit.list_feedback(request_id)


@router.post(
    "/audits/{request_id}/feedback",
    response_model=FeedbackRecord,
    status_code=status.HTTP_201_CREATED,
    summary="Record a reviewer's agreement or override",
)
async def post_feedback(
    request_id: str, body: FeedbackRequest, audit: AuditDep
) -> FeedbackRecord:
    if await audit.get(request_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cannot attach feedback: no audit record for request '{request_id}'.",
        )
    record = FeedbackRecord(request_id=request_id, **body.model_dump())
    await audit.save_feedback(record)
    return record


__all__ = ["AuditPage", "FeedbackRequest", "router"]
