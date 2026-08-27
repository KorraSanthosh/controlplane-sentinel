"""API error contracts.

The rule applied throughout: **a failure of the control plane must not be reported as a clean
result.** Concretely —

* the model is unreachable → 503. There is no response, so there is nothing to govern and no
  audit record to write. Returning a synthesised answer with ``decision=ALLOW`` would be a lie
  told by the exact component whose job is to prevent them.
* the policy is unloadable → 500. Serving traffic with no policy is worse than serving none.
* the audit store is unreachable → **not** an error. The decision was still made and returned;
  the gap is logged at ERROR and reported by ``/health``. That trade is documented in
  :mod:`app.services.audit.service`.

Every handler emits the same JSON shape so the dashboard has one error path to render.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.services.llm.base import LLMUnavailable
from app.services.policy.conditions import PolicyError

logger = logging.getLogger(__name__)


def error_body(code: str, message: str, detail: str | None = None) -> dict[str, object]:
    body: dict[str, object] = {"error": {"code": code, "message": message}}
    if detail:
        body["error"]["detail"] = detail  # type: ignore[index]
    return body


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(LLMUnavailable)
    async def _llm_unavailable(request: Request, exc: LLMUnavailable) -> JSONResponse:
        logger.error("LLM provider unavailable for %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=error_body(
                "llm_unavailable",
                "The language model could not be reached, so no response was generated and "
                "nothing was evaluated. No decision was made and no audit record was written.",
                str(exc),
            ),
        )

    @app.exception_handler(PolicyError)
    async def _policy_error(request: Request, exc: PolicyError) -> JSONResponse:
        logger.error("Policy error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body(
                "policy_error",
                "The active governance policy could not be applied. The request was refused "
                "rather than evaluated without a policy.",
                str(exc),
            ),
        )


__all__ = ["error_body", "register_error_handlers"]
