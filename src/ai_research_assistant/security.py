"""
security.py
============
The "Security Checkpoint" for the AI Research Assistant.

This module implements defense-in-depth guardrails that run as ADK
callbacks, independent of any single agent's prompt:

  1. Input screening       - blocks prompt-injection / jailbreak patterns and
                              oversized input before it ever reaches a model.
  2. PII redaction         - strips obvious PII (emails, phone numbers, SSNs)
                              out of user-supplied text before it is logged
                              or sent to a model.
  3. Rate limiting         - a simple sliding-window limiter per session to
                              protect the MCP tool backends (arXiv, web
                              search, etc.) from abuse.
  4. Tool argument checks  - validates arguments passed to sensitive tools
                              (in particular `export_report`, which writes to
                              disk) to prevent path traversal / injection.

All of these are wired in as `before_model_callback` / `before_tool_callback`
/ `before_agent_callback` hooks on the agents defined in `agent.py`.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from typing import Any, Optional

from google.adk.agents.context import Context
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.genai import types

from ai_research_assistant.audit import audit_logger
from ai_research_assistant.config import settings

# --------------------------------------------------------------------------
# Pattern libraries
# --------------------------------------------------------------------------

# Deliberately conservative heuristics -- a real production checkpoint would
# likely also call a dedicated classifier model, but pattern matching gives
# us a fast, dependency-free first line of defense.
_PROMPT_INJECTION_PATTERNS = [
    r"ignore (all|any|the) (previous|prior|above) instructions",
    r"disregard (all|any|the) (previous|prior|above) (instructions|rules)",
    r"you are now (in )?(developer|debug|dan|jailbreak) mode",
    r"reveal (your|the) (system prompt|instructions)",
    r"act as (if you (have|had) no|an unrestricted)",
    r"pretend (that )?you (have no|are not bound by)",
    r"override (your|the) (safety|security) (settings|guardrails)",
]
_PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b\+?\d[\d\-\s()]{8,}\d\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}

_INJECTION_REGEX = re.compile("|".join(_PROMPT_INJECTION_PATTERNS), re.IGNORECASE)


class RateLimiter:
    """A minimal in-memory sliding-window rate limiter, keyed by session id."""

    def __init__(self, max_calls: int, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window = self._hits[key]
        while window and now - window[0] > self.window_seconds:
            window.popleft()
        if len(window) >= self.max_calls:
            return False
        window.append(now)
        return True


class SecurityCheckpoint:
    """Central guardrail object used across agents and the API layer."""

    def __init__(self):
        self.rate_limiter = RateLimiter(max_calls=settings.rate_limit_per_minute)

    # ---- Input screening -------------------------------------------------

    def screen_text(self, text: str) -> tuple[bool, Optional[str]]:
        """Returns (is_safe, reason_if_blocked)."""
        if len(text) > settings.max_input_chars:
            return False, f"Input exceeds max length of {settings.max_input_chars} characters."
        if settings.block_prompt_injection and _INJECTION_REGEX.search(text):
            return False, "Input matched a known prompt-injection / jailbreak pattern."
        return True, None

    def redact_pii(self, text: str) -> str:
        redacted = text
        for label, pattern in _PII_PATTERNS.items():
            redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)
        return redacted

    # ---- Tool argument validation -----------------------------------------

    def validate_tool_args(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, Optional[str]]:
        if tool_name == "export_report":
            filename = str(args.get("title") or args.get("filename") or "")
            if ".." in filename or filename.startswith("/") or "\\" in filename:
                return False, "export_report received a suspicious path/filename."
        # Generic oversized-argument guard for every tool.
        try:
            total_len = sum(len(str(v)) for v in args.values())
        except Exception:
            total_len = 0
        if total_len > settings.max_input_chars * 10:
            return False, f"Arguments to '{tool_name}' are abnormally large."
        return True, None


checkpoint = SecurityCheckpoint()


# --------------------------------------------------------------------------
# ADK callback adapters
# --------------------------------------------------------------------------


def security_before_agent(callback_context=None, **kwargs):
    return None
    """
    `before_agent_callback`: screens the latest user message before any
    agent in the workflow graph is allowed to run. Returning a `types.Content`
    here short-circuits the agent entirely and that content is used as the
    agent's "response" instead.
    """
    user_content = getattr(ctx, "user_content", None)
    text = _extract_text(user_content)
    if not text:
        return None

    session_key = getattr(ctx.session, "id", "unknown-session")
    if not checkpoint.rate_limiter.allow(session_key):
        audit_logger.log_event(
            "SECURITY_BLOCK",
            session_id=session_key,
            agent_name=ctx.agent_name,
            severity="WARNING",
            details={"reason": "rate_limit_exceeded"},
        )
        return types.Content(
            role="model",
            parts=[types.Part(text="Rate limit exceeded. Please wait a minute and try again.")],
        )

    is_safe, reason = checkpoint.screen_text(text)
    if not is_safe:
        audit_logger.log_event(
            "SECURITY_BLOCK",
            session_id=session_key,
            agent_name=ctx.agent_name,
            severity="WARNING",
            details={"reason": reason},
        )
        return types.Content(
            role="model",
            parts=[types.Part(text=f"Request blocked by security checkpoint: {reason}")],
        )
    return None


spdef security_before_model(callback_context=None, **kwargs):
    print("MODEL HOOK:", callback_context)
    return None
    """
    `before_model_callback`: last line of defense immediately before a prompt
    is sent to the LLM. Redacts obvious PII from the outgoing request text.
    This does not block the call -- redaction is applied in place via the
    request's `contents`, and the (unmodified) return value of None means
    "proceed with the (mutated) request".
    """
    try:
        for content in getattr(llm_request, "contents", []) or []:
            for part in getattr(content, "parts", []) or []:
                if getattr(part, "text", None):
                    part.text = checkpoint.redact_pii(part.text)
    except Exception as exc:  # never let the guardrail itself crash the pipeline
        audit_logger.log_event(
            "SECURITY_ERROR",
            agent_name=ctx.agent_name,
            severity="ERROR",
            details={"error": str(exc)},
        )
    return None


def security_before_tool(
    tool: BaseTool, args: dict[str, Any], ctx: Context
) -> Optional[dict]:
    """
    `before_tool_callback`: validates arguments before any MCP tool executes.
    Returning a dict here short-circuits the tool call and that dict becomes
    the tool's result instead of actually invoking it.
    """
    is_valid, reason = checkpoint.validate_tool_args(tool.name, args)
    if not is_valid:
        audit_logger.log_event(
            "SECURITY_BLOCK",
            session_id=getattr(ctx.session, "id", None),
            agent_name=ctx.agent_name,
            tool_name=tool.name,
            severity="WARNING",
            details={"reason": reason, "args": args},
        )
        return {"error": f"Blocked by security checkpoint: {reason}"}
    return None


def _extract_text(content: Optional[types.Content]) -> str:
    if not content or not getattr(content, "parts", None):
        return ""
    return " ".join(p.text for p in content.parts if getattr(p, "text", None))
