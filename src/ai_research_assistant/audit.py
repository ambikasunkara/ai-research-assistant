"""
audit.py
========
Structured, append-only audit logging for every agent invocation, tool call,
and model call in the system.

Design goals:
  * Every audit record is a single JSON line (JSONL) written to
    `settings.audit_log_path`, making it trivial to ship to a SIEM later.
  * The functions in this module are registered as ADK lifecycle callbacks
    (`before_agent_callback`, `after_agent_callback`, `before_tool_callback`,
    `after_tool_callback`, `before_model_callback`) so that logging happens
    automatically for every agent in the workflow graph without agents having
    to remember to log anything themselves.
  * Callbacks never raise -- a failure to log must never break the research
    workflow.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from google.adk.agents.context import Context
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.genai import types

from ai_research_assistant.config import settings

# --------------------------------------------------------------------------
# Low-level logger: writes newline-delimited JSON audit records to disk and
# mirrors a human-readable line to stdout via the standard logging module.
# --------------------------------------------------------------------------

_logger = logging.getLogger("ai_research_assistant.audit")
_logger.setLevel(settings.log_level)
if not _logger.handlers:
    _stream_handler = logging.StreamHandler()
    _stream_handler.setFormatter(
        logging.Formatter("%(asctime)s [AUDIT] %(message)s")
    )
    _logger.addHandler(_stream_handler)


class AuditLogger:
    """Writes structured audit events to a JSONL file plus stdout."""

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or str(settings.audit_log_path)

    def log_event(
        self,
        event_type: str,
        *,
        session_id: Optional[str] = None,
        invocation_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        tool_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        severity: str = "INFO",
    ) -> None:
        """Append one structured audit record."""
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "severity": severity,
            "session_id": session_id,
            "invocation_id": invocation_id,
            "agent_name": agent_name,
            "tool_name": tool_name,
            "details": details or {},
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:  # never let audit failures break the workflow
            _logger.error("Failed to write audit record to %s: %s", self.log_path, exc)

        _logger.info(
            "%s | agent=%s tool=%s session=%s :: %s",
            event_type,
            agent_name,
            tool_name,
            session_id,
            json.dumps(details or {}, default=str)[:500],
        )


audit_logger = AuditLogger()


# --------------------------------------------------------------------------
# ADK lifecycle callbacks
#
# These functions match the EXACT keyword-only calling convention ADK uses
# to invoke lifecycle callbacks (verified against google-adk 2.3.x source in
# google/adk/agents/base_agent.py, google/adk/flows/llm_flows/base_llm_flow.py,
# and google/adk/flows/llm_flows/functions.py):
#
#   before_agent_callback(callback_context=...)            -> Optional[Content]
#   after_agent_callback(callback_context=...)              -> Optional[Content]
#   before_tool_callback(tool=..., args=..., tool_context=...)      -> Optional[dict]
#   after_tool_callback(tool=..., args=..., tool_context=..., tool_response=...) -> Optional[dict]
#   before_model_callback(callback_context=..., llm_request=...)    -> Optional[LlmResponse]
#
# All calls are keyword-only, so the parameter *names* below are not
# cosmetic -- getting them wrong raises `TypeError: unexpected keyword
# argument` at runtime the moment ADK invokes the callback. `ToolContext`
# and `CallbackContext` are both aliases of the same underlying `Context`
# class in this ADK version, so the attribute access below (`.state`,
# `.session`, `.agent_name`, `.invocation_id`) is identical either way.
# --------------------------------------------------------------------------

def audit_before_agent(callback_context: Context, **kwargs) -> Optional[types.Content]:
    """Logs the start of every agent invocation and stamps a start time in state."""
    ctx = callback_context
    ctx.state[f"_audit_start_time::{ctx.agent_name}"] = time.monotonic()
    audit_logger.log_event(
        "AGENT_START",
        session_id=getattr(ctx.session, "id", None),
        invocation_id=ctx.invocation_id,
        agent_name=ctx.agent_name,
        details={"user_id": getattr(ctx, "user_id", None)},
    )
    return None  # returning None lets the agent execute normally


def audit_after_agent(callback_context: Context, **kwargs) -> Optional[types.Content]:
    """Logs completion of every agent invocation, including elapsed time."""
    ctx = callback_context
    start = ctx.state.get(f"_audit_start_time::{ctx.agent_name}")
    elapsed_ms = round((time.monotonic() - start) * 1000, 2) if start else None
    audit_logger.log_event(
        "AGENT_END",
        session_id=getattr(ctx.session, "id", None),
        invocation_id=ctx.invocation_id,
        agent_name=ctx.agent_name,
        details={"elapsed_ms": elapsed_ms},
    )
    return None


def audit_before_tool(
    tool: BaseTool, args: dict[str, Any], tool_context: Context, **kwargs
) -> Optional[dict]:
    """Logs every tool invocation *before* it runs, including its arguments."""
    ctx = tool_context
    audit_logger.log_event(
        "TOOL_CALL_START",
        session_id=getattr(ctx.session, "id", None),
        invocation_id=ctx.invocation_id,
        agent_name=ctx.agent_name,
        tool_name=tool.name,
        details={"args": _safe_truncate(args)},
    )
    return None  # returning None allows the tool call to proceed unmodified


def audit_after_tool(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: Context,
    tool_response: dict,
    **kwargs,
) -> Optional[dict]:
    """Logs the result of every tool invocation."""
    ctx = tool_context
    audit_logger.log_event(
        "TOOL_CALL_END",
        session_id=getattr(ctx.session, "id", None),
        invocation_id=ctx.invocation_id,
        agent_name=ctx.agent_name,
        tool_name=tool.name,
        details={"response_preview": _safe_truncate(tool_response)},
    )
    return None  # do not modify the tool response


def audit_before_model(callback_context: Context, llm_request: LlmRequest, **kwargs) -> None:
    """Logs the fact that a model call is about to be made (no payload dump)."""
    ctx = callback_context
    audit_logger.log_event(
        "MODEL_CALL",
        session_id=getattr(ctx.session, "id", None),
        invocation_id=ctx.invocation_id,
        agent_name=ctx.agent_name,
        details={"model": getattr(llm_request, "model", "unknown")},
    )
    return None


def _safe_truncate(payload: Any, limit: int = 800) -> Any:
    """Prevents huge payloads (e.g. full paper text) from bloating the audit log."""
    try:
        text = json.dumps(payload, default=str)
    except (TypeError, ValueError):
        text = str(payload)
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return payload
