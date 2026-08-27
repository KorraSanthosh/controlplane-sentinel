"""``POST /chat`` — the governed generation endpoint (FR-01).

The whole product, in one call: generate, evaluate, decide, act, audit.

Persistence runs in a ``BackgroundTask`` rather than inline. The decision has already been made
and is already in the response body, so making the caller wait for a database write — or fail
because of one — would put the governance layer's storage on the critical path of the product it
protects. The gap is logged at ERROR and reported by ``/health``; the trade is documented in
:mod:`app.services.audit.service`.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks

from app.api.deps import AuditDep, OrchestratorDep
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Generate a response and evaluate it through ControlPlane",
)
async def chat(
    request: ChatRequest,
    background: BackgroundTasks,
    orchestrator: OrchestratorDep,
    audit: AuditDep,
) -> ChatResponse:
    response, record = await orchestrator.process(request)
    background.add_task(audit.persist, record)
    return response


__all__ = ["router"]
