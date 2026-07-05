"""
fast_api_app.py
================
FastAPI wrapper around the AI Research Assistant's ADK `Runner`.

Endpoints
---------
  GET  /health              - liveness check
  POST /research             - starts (or continues) a research workflow run
  POST /approve               - resumes a workflow paused at the Human Approval
                                gate (`request_input`) with the human's decision
  GET  /reports/{filename}   - fetch a previously exported report file

Human-in-the-loop resumption
-----------------------------
The `human_approval_agent` in `agent.py` calls ADK's built-in `request_input`
long-running tool. When that happens, `runner.run_async()` yields an `Event`
whose `long_running_tool_ids` set contains the pending function call's id,
and the run stops advancing until a matching `FunctionResponse` is supplied
in a follow-up call. `POST /approve` builds that `FunctionResponse` (using
the same `name`/`id` contract ADK's `request_input` tool relies on) and
resumes the same session.

Security & audit
------------------
Every request must present a valid `X-API-Key` header (checked against
`settings.research_api_keys_set` -- this is the API-layer half of the
project's Security Checkpoint; the agent-layer half lives in `security.py`).
Every request is also recorded via `audit_logger`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from ai_research_assistant.agent import root_agent
from ai_research_assistant.audit import audit_logger
from ai_research_assistant.config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("ai_research_assistant.api")

# --------------------------------------------------------------------------
# ADK runtime wiring
# --------------------------------------------------------------------------
# `InMemorySessionService` is fine for local development / the Kaggle demo.
# For a real deployment, swap in `DatabaseSessionService` or
# `VertexAiSessionService` so sessions (and pending approvals) survive
# process restarts and work across multiple API workers.
session_service = InMemorySessionService()
runner = Runner(
    app_name=settings.app_name,
    agent=root_agent,
    session_service=session_service,
)

# The literal function name ADK's built-in `request_input` long-running tool
# is registered under (see `google.adk.tools._request_input_tool`). Resuming
# a paused run requires echoing this exact name back in the FunctionResponse.
REQUEST_INPUT_FUNCTION_NAME = "adk_request_input"

# In-memory store of pending human-approval requests, keyed by session_id.
# NOTE: like `InMemorySessionService`, this is process-local. A production
# deployment should persist this alongside session state (e.g. in the same
# database backing a `DatabaseSessionService`).
_pending_approvals: dict[str, dict[str, Any]] = {}


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------
class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=settings.max_input_chars)
    user_id: str = Field(default="anonymous")
    session_id: Optional[str] = Field(default=None)


class ApprovalRequest(BaseModel):
    session_id: str
    user_id: str = Field(default="anonymous")
    approved: bool
    comments: Optional[str] = None


class WorkflowResponse(BaseModel):
    session_id: str
    status: str  # "completed" | "pending_approval" | "error"
    message: str
    state: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------
app = FastAPI(
    title="AI Research Assistant API",
    description="Multi-agent research automation system built on Google ADK.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


async def verify_api_key(x_api_key: str = Header(default="")) -> str:
    """Security Checkpoint (API layer): every request must carry a valid API key."""
    if x_api_key not in settings.research_api_keys_set:
        audit_logger.log_event(
            "SECURITY_BLOCK",
            severity="WARNING",
            details={"reason": "invalid_or_missing_api_key"},
        )
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")
    return x_api_key


@app.middleware("http")
async def audit_requests(request: Request, call_next):
    """Audit-logs every inbound HTTP request and its resulting status code."""
    audit_logger.log_event(
        "HTTP_REQUEST",
        details={"method": request.method, "path": request.url.path},
    )
    response = await call_next(request)
    audit_logger.log_event(
        "HTTP_RESPONSE",
        details={"path": request.url.path, "status_code": response.status_code},
    )
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@app.post("/research", response_model=WorkflowResponse)
async def start_research(
    payload: ResearchRequest, _api_key: str = Depends(verify_api_key)
) -> WorkflowResponse:
    """Starts a new research workflow run for the given query."""
    session_id = payload.session_id or f"session-{payload.user_id}-{id(payload)}"

    existing = await session_service.get_session(
        app_name=settings.app_name, user_id=payload.user_id, session_id=session_id
    )
    if existing is None:
        await session_service.create_session(
            app_name=settings.app_name, user_id=payload.user_id, session_id=session_id
        )

    new_message = types.Content(role="user", parts=[types.Part(text=payload.query)])
    return await _run_and_collect(payload.user_id, session_id, new_message)


@app.post("/approve", response_model=WorkflowResponse)
async def approve_research(
    payload: ApprovalRequest, _api_key: str = Depends(verify_api_key)
) -> WorkflowResponse:
    """Resumes a workflow paused at the Human Approval gate."""
    pending = _pending_approvals.pop(payload.session_id, None)
    if pending is None:
        raise HTTPException(
            status_code=404,
            detail="No pending approval found for this session_id.",
        )

    audit_logger.log_event(
        "HUMAN_APPROVAL_DECISION",
        session_id=payload.session_id,
        details={"approved": payload.approved, "comments": payload.comments},
    )

    function_response_part = types.Part(
        function_response=types.FunctionResponse(
            id=pending["call_id"],
            name=REQUEST_INPUT_FUNCTION_NAME,
            response={"output": payload.approved, "comments": payload.comments or ""},
        )
    )
    new_message = types.Content(role="user", parts=[function_response_part])
    return await _run_and_collect(payload.user_id, payload.session_id, new_message)


@app.get("/reports/{filename}")
async def get_report(filename: str, _api_key: str = Depends(verify_api_key)) -> FileResponse:
    """Serves a previously exported report file by name."""
    safe_name = Path(filename).name  # strip any path components defensively
    file_path = (settings.reports_dir / safe_name).resolve()
    if settings.reports_dir.resolve() not in file_path.parents or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(file_path)


# --------------------------------------------------------------------------
# Core run/resume helper
# --------------------------------------------------------------------------
async def _run_and_collect(
    user_id: str, session_id: str, new_message: types.Content
) -> WorkflowResponse:
    """Drives the runner until either the workflow completes or pauses for approval."""
    last_text = ""
    pending_call_id: Optional[str] = None
    pending_message = ""

    try:
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=new_message
        ):
            if event.long_running_tool_ids:
                for part in (event.content.parts if event.content else []):
                    fc = getattr(part, "function_call", None)
                    if fc and fc.id in event.long_running_tool_ids:
                        pending_call_id = fc.id
                        pending_message = (fc.args or {}).get("message", "Approval required.")

            if event.content and event.content.parts:
                text_parts = [p.text for p in event.content.parts if getattr(p, "text", None)]
                if text_parts:
                    last_text = " ".join(text_parts)

    except Exception as exc:
        logger.exception("Workflow run failed for session %s", session_id)
        audit_logger.log_event(
            "WORKFLOW_ERROR",
            session_id=session_id,
            severity="ERROR",
            details={"error": str(exc)},
        )
        return WorkflowResponse(
            session_id=session_id, status="error", message=f"Workflow failed: {exc}", state={}
        )

    session = await session_service.get_session(
        app_name=settings.app_name, user_id=user_id, session_id=session_id
    )
    state_snapshot = dict(session.state) if session else {}

    if pending_call_id:
        _pending_approvals[session_id] = {"call_id": pending_call_id, "message": pending_message}
        audit_logger.log_event(
            "WORKFLOW_PAUSED_FOR_APPROVAL", session_id=session_id, details={"prompt": pending_message}
        )
        return WorkflowResponse(
            session_id=session_id,
            status="pending_approval",
            message=pending_message,
            state=state_snapshot,
        )

    return WorkflowResponse(
        session_id=session_id,
        status="completed",
        message=last_text or "Workflow completed.",
        state=state_snapshot,
    )


def main() -> None:
    """Entrypoint for `research-api` console script / `python -m ...fast_api_app`."""
    import uvicorn

    uvicorn.run(
        "ai_research_assistant.fast_api_app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
