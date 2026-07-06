"""
agent.py
========
The complete multi-agent workflow graph for the AI Research Assistant,
built on Google ADK 2.x.

Architecture (a `SequentialAgent` workflow graph)
--------------------------------------------------

    orchestrator_agent (SequentialAgent)
        1. research_agent            (LlmAgent)
               -> uses paper_search_agent (LlmAgent) as an AgentTool
               -> uses the `search_web` MCP tool directly
        2. summarization_agent       (LlmAgent)
        3. fact_verification_agent   (LlmAgent)   -- MCP: fact_check, compare_sources
        4. human_approval_agent      (LlmAgent)   -- ADK `request_input` (human-in-the-loop)
        5. citation_agent            (LlmAgent)   -- MCP: generate_citations
        6. report_generator_agent    (LlmAgent)   -- MCP: export_report

Cross-cutting concerns (Security Checkpoint + Audit Logging) are wired in as
ADK lifecycle callbacks (`before_agent_callback`, `before_model_callback`,
`before_tool_callback`, `after_tool_callback`, `after_agent_callback`) on
every agent, so they apply uniformly across the whole graph without any
single agent having to remember to invoke them.

Data flows between agents exclusively through `ctx.state` (session state):
each agent writes its result under `output_key`, and downstream agents read
those values back via `{state_key}` instruction-template interpolation,
which ADK resolves automatically from session state before each model call.

`root_agent` at the bottom is the conventional entry point ADK's CLI/web UI
and `Runner` look for.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from google.adk.agents.context import Context
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools import request_input
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from mcp import StdioServerParameters
from google.genai import types

from ai_research_assistant.audit import (
    audit_after_agent,
    audit_after_tool,
    audit_before_agent,
    audit_before_model,
    audit_before_tool,
)
from ai_research_assistant.config import settings
from ai_research_assistant.security import (
    security_before_agent,
    security_before_model,
    security_before_tool,
)

logger = logging.getLogger("ai_research_assistant.agent")

# ==========================================================================
# 1. MCP Server connection
#
# The MCP server (mcp_server.py) is launched as a local subprocess over
# stdio, exposing our six research tools. Each `McpToolset` below connects
# to its own instance of that subprocess and exposes a filtered subset of
# its tools to a specific agent -- this keeps each agent's tool surface
# minimal and auditable (principle of least privilege).
# ==========================================================================


def _mcp_connection() -> StdioConnectionParams:
    """Builds the stdio connection parameters for launching the MCP server."""
    return StdioConnectionParams(
        server_params=StdioServerParameters(
            command=settings.mcp_server_command,
            args=["-m", settings.mcp_server_module],
        ),
        timeout=30.0,
    )


paper_search_tools = McpToolset(
    connection_params=_mcp_connection(),
    tool_filter=["search_papers"],
    tool_name_prefix="mcp",
)

web_search_tools = McpToolset(
    connection_params=_mcp_connection(),
    tool_filter=["search_web"],
    tool_name_prefix="mcp",
)

verification_tools = McpToolset(
    connection_params=_mcp_connection(),
    tool_filter=["fact_check", "compare_sources"],
    tool_name_prefix="mcp",
)

citation_tools = McpToolset(
    connection_params=_mcp_connection(),
    tool_filter=["generate_citations"],
    tool_name_prefix="mcp",
)

report_tools = McpToolset(
    connection_params=_mcp_connection(),
    tool_filter=["export_report"],
    tool_name_prefix="mcp",
)


# ==========================================================================
# 2. Cross-cutting callback bundles (Security Checkpoint + Audit Logging)
#
# Every LlmAgent below is wired with the same four hook points. Order
# matters: security checkpoints run before audit logging on the "before"
# side, so that blocked requests are still recorded, but blocked *content*
# never reaches the model.
# ==========================================================================

BEFORE_AGENT_HOOKS = [security_before_agent, audit_before_agent]
AFTER_AGENT_HOOKS = [audit_after_agent]
BEFORE_MODEL_HOOKS = [security_before_model, audit_before_model]
BEFORE_TOOL_HOOKS = [security_before_tool, audit_before_tool]
AFTER_TOOL_HOOKS = [audit_after_tool]


def initialize_workflow_state(callback_context: Context, **kwargs) -> Optional[types.Content]:
    """
    `before_agent_callback` registered on the top-level orchestrator only.

    Demonstrates explicit `ctx.state` usage: seeds the shared session state
    with the original research query, a workflow id, and a start timestamp
    *before* any sub-agent runs, so every downstream agent (and the audit
    log) can reference `{research_query}` / `{workflow_id}` consistently.

    NOTE: like every other callback in this project, ADK invokes this as
    `callback(callback_context=...)` -- keyword-only -- so the parameter
    must be named `callback_context`, not `ctx`.
    """
    ctx = callback_context
    if "workflow_id" in ctx.state:
        # Already initialized (e.g. this is a resumed/human-approval turn).
        return None

    user_text = ""
    if ctx.user_content and ctx.user_content.parts:
        user_text = " ".join(p.text for p in ctx.user_content.parts if getattr(p, "text", None))

    ctx.state["workflow_id"] = str(uuid.uuid4())
    ctx.state["research_query"] = user_text.strip()
    ctx.state["started_at"] = datetime.now(timezone.utc).isoformat()
    logger.info("Initialized workflow %s for query: %s", ctx.state["workflow_id"], user_text[:120])
    return None


# ==========================================================================
# 3. Paper Search Agent
#
# A focused, narrow-scope agent whose only job is to query arXiv via the
# `search_papers` MCP tool. It is invoked by the Research Agent as an
# AgentTool (see below) rather than being reachable directly by the user.
# ==========================================================================

paper_search_agent = LlmAgent(
    name="paper_search_agent",
    model=settings.model_flash,
    description="Searches arXiv for academic papers relevant to a research query.",
    instruction=(
        "You are the Paper Search Agent. Your ONLY job is to find relevant "
        "academic papers for the research topic: '{research_query}'.\n\n"
        "1. Call the `search_papers` tool with a well-formed query derived from "
        "the topic (use precise academic keywords, not the raw user sentence).\n"
        "2. Return a concise structured list of the papers found: title, authors, "
        "year, and a one-sentence description of relevance.\n"
        "Do not fabricate papers. If the tool returns no results, say so plainly."
    ),
    tools=[paper_search_tools],
    output_key="paper_search_results",
    before_agent_callback=BEFORE_AGENT_HOOKS,
    after_agent_callback=AFTER_AGENT_HOOKS,
    before_model_callback=BEFORE_MODEL_HOOKS,
    before_tool_callback=BEFORE_TOOL_HOOKS,
    after_tool_callback=AFTER_TOOL_HOOKS,
)


# ==========================================================================
# 4. Research Agent
#
# Orchestrates the broader research step: delegates academic search to
# `paper_search_agent` via `AgentTool`, and independently uses the
# `search_web` MCP tool for background / non-academic context. Combines
# both into a single `research_findings` artifact in session state.
# ==========================================================================

research_agent = LlmAgent(
    name="research_agent",
    model=settings.model_pro,
    description="Gathers academic and web research for a given topic.",
    instruction=(
        "You are the Research Agent for topic: '{research_query}'.\n\n"
        "Steps:\n"
        "1. Call the `paper_search_agent` tool (an AgentTool wrapping a "
        "specialist sub-agent) to gather relevant academic papers.\n"
        "2. Call the `search_web` tool directly to gather relevant background "
        "context, definitions, or recent news not covered by academic papers.\n"
        "3. Synthesize both into a single organized research brief covering: "
        "key concepts, the current state of the art, notable papers (with URLs), "
        "and open questions.\n\n"
        "Be thorough but avoid duplicating raw tool output verbatim -- synthesize."
    ),
    tools=[
        AgentTool(agent=paper_search_agent),
        web_search_tools,
    ],
    output_key="research_findings",
    before_agent_callback=BEFORE_AGENT_HOOKS,
    after_agent_callback=AFTER_AGENT_HOOKS,
    before_model_callback=BEFORE_MODEL_HOOKS,
    before_tool_callback=BEFORE_TOOL_HOOKS,
    after_tool_callback=AFTER_TOOL_HOOKS,
)


# ==========================================================================
# 5. Summarization Agent
#
# Pure-reasoning agent (no tools): condenses `research_findings` into a
# tight, decision-ready summary. Reads state via instruction interpolation.
# ==========================================================================

summarization_agent = LlmAgent(
    name="summarization_agent",
    model=settings.model_flash,
    description="Summarizes the research findings into a concise brief.",
    instruction=(
        "You are the Summarization Agent. Summarize the following research "
        "findings into a clear, well-structured summary (max ~350 words) with "
        "these sections: Overview, Key Findings, Notable Papers, Open Questions.\n\n"
        "Research findings:\n"
        "{research_findings}"
    ),
    tools=[],
    output_key="summary",
    before_agent_callback=BEFORE_AGENT_HOOKS,
    after_agent_callback=AFTER_AGENT_HOOKS,
    before_model_callback=BEFORE_MODEL_HOOKS,
)


# ==========================================================================
# 6. Fact Verification Agent
#
# Uses the `fact_check` and `compare_sources` MCP tools to ground its
# judgment in lexical evidence rather than relying purely on the model's
# parametric knowledge, then produces a verification report with a
# confidence rating per key claim.
# ==========================================================================

fact_verification_agent = LlmAgent(
    name="fact_verification_agent",
    model=settings.model_pro,
    description="Verifies key claims in the research summary against sources.",
    instruction=(
        "You are the Fact Verification Agent. Given the summary below, extract "
        "the 3-5 most load-bearing factual claims and verify each one:\n\n"
        "1. For each claim, call the `fact_check` tool, passing the claim text "
        "and the list of source papers/results you have available from earlier "
        "steps (reconstruct a minimal source list of dicts with 'title' and "
        "'summary' fields from the research findings below).\n"
        "2. Optionally call `compare_sources` if multiple sources disagree, to "
        "understand where the overlap/divergence lies.\n"
        "3. Produce a verification report: for each claim, state the claim, the "
        "confidence score, whether it is well-supported, and any caveats.\n\n"
        "Summary to verify:\n{summary}\n\n"
        "Research findings (source material):\n{research_findings}"
    ),
    tools=[verification_tools],
    output_key="verification_report",
    before_agent_callback=BEFORE_AGENT_HOOKS,
    after_agent_callback=AFTER_AGENT_HOOKS,
    before_model_callback=BEFORE_MODEL_HOOKS,
    before_tool_callback=BEFORE_TOOL_HOOKS,
    after_tool_callback=AFTER_TOOL_HOOKS,
)


# ==========================================================================
# 7. Human Approval Agent  (Human-in-the-loop via ADK's `request_input`)
#
# `request_input` is ADK's built-in LongRunningFunctionTool for pausing a
# workflow to ask the human operator a question. When this agent calls it,
# the ADK runtime emits an event with `long_running_tool_ids` populated and
# the run yields control back to the caller (see fast_api_app.py's
# `/approve` endpoint for how the paused run is resumed with the human's
# decision).
# ==========================================================================

human_approval_agent = LlmAgent(
    name="human_approval_agent",
    model=settings.model_flash,
    description="Pauses the workflow to obtain explicit human sign-off before publishing.",
    instruction=(
        "You are the Human Approval gate. A human researcher must approve the "
        "findings before a citation list and final report are generated.\n\n"
        "Call the `request_input` tool exactly once with:\n"
        "  message: a concise (<150 word) recap of the summary and verification "
        "confidence, ending with 'Approve publishing the final report? (yes/no)'\n"
        '  response_schema: {"type": "boolean"}\n\n'
        "Summary:\n{summary}\n\n"
        "Verification report:\n{verification_report}\n\n"
        "After the human responds, output exactly one word: 'approved' if they "
        "said yes/true, otherwise 'rejected'. Do not add any other text."
    ),
    tools=[request_input],
    output_key="human_approval_decision",
    before_agent_callback=BEFORE_AGENT_HOOKS,
    after_agent_callback=AFTER_AGENT_HOOKS,
    before_model_callback=BEFORE_MODEL_HOOKS,
    before_tool_callback=BEFORE_TOOL_HOOKS,
    after_tool_callback=AFTER_TOOL_HOOKS,
)


# ==========================================================================
# 8. Citation Agent
#
# Only runs meaningfully once approval has been granted (the Report
# Generator Agent enforces this gate), but citations are pre-generated here
# so the final report assembly step is a simple compilation.
# ==========================================================================

citation_agent = LlmAgent(
    name="citation_agent",
    model=settings.model_flash,
    description="Generates formatted bibliographic citations for the cited papers.",
    instruction=(
        "You are the Citation Agent. Extract the list of academic papers "
        "referenced in the research findings below (title, authors, published "
        "date, url) and call the `generate_citations` tool with style='APA' to "
        "produce a formatted bibliography. Return the bibliography as a "
        "numbered list.\n\n"
        "Research findings:\n{research_findings}"
    ),
    tools=[citation_tools],
    output_key="citations",
    before_agent_callback=BEFORE_AGENT_HOOKS,
    after_agent_callback=AFTER_AGENT_HOOKS,
    before_model_callback=BEFORE_MODEL_HOOKS,
    before_tool_callback=BEFORE_TOOL_HOOKS,
    after_tool_callback=AFTER_TOOL_HOOKS,
)


# ==========================================================================
# 9. Report Generator Agent
#
# Compiles the summary, verification report, and citations into a final
# Markdown report and calls `export_report` to persist it to disk -- but
# only if `human_approval_decision` is 'approved'. This is the terminal
# node of the workflow graph.
# ==========================================================================

report_generator_agent = LlmAgent(
    name="report_generator_agent",
    model=settings.model_pro,
    description="Compiles the final research report and exports it to disk.",
    instruction=(
        "You are the Report Generator Agent for workflow '{workflow_id}' on "
        "topic '{research_query}'.\n\n"
        "The human approval decision was: {human_approval_decision}\n\n"
        "If the decision is 'rejected', do NOT call any tool. Respond with a "
        "short message explaining that the human reviewer declined to approve "
        "the report, and no file was created.\n\n"
        "If the decision is 'approved':\n"
        "1. Compile a complete Markdown report with sections: Title, Executive "
        "Summary, Key Findings, Fact Verification, References.\n"
        "   - Executive Summary + Key Findings come from: {summary}\n"
        "   - Fact Verification comes from: {verification_report}\n"
        "   - References come from: {citations}\n"
        "2. Call the `export_report` tool with an appropriate title, the full "
        "compiled Markdown content, and format='markdown'.\n"
        "3. Confirm the file path returned by the tool to the user."
    ),
    tools=[report_tools],
    output_key="final_report",
    before_agent_callback=BEFORE_AGENT_HOOKS,
    after_agent_callback=AFTER_AGENT_HOOKS,
    before_model_callback=BEFORE_MODEL_HOOKS,
    before_tool_callback=BEFORE_TOOL_HOOKS,
    after_tool_callback=AFTER_TOOL_HOOKS,
)


# ==========================================================================
# 10. Orchestrator Agent (ADK Workflow Graph)
#
# A `SequentialAgent` chains the sub-agents in a fixed pipeline. This is the
# ADK "workflow graph" for this project: research -> summarize -> verify ->
# human approval -> cite -> report. Every edge in this graph is `ctx.state`:
# each node reads what previous nodes wrote via output_key.
#
# NOTE: `google-adk` has begun surfacing a newer `Workflow`/graph-node API
# alongside the classic `SequentialAgent`/`LlmAgent` model used throughout
# this file, and `SequentialAgent` is marked deprecated (but fully
# functional) as of google-adk 2.3.x. This project intentionally targets
# the classic, stable `LlmAgent` + `SequentialAgent` + callback API for the
# capstone, since it is broadly documented and battle-tested; migrating to
# `Workflow` is a natural next step and is called out in README.md under
# "Roadmap".
# ==========================================================================

orchestrator_agent = SequentialAgent(
    name="orchestrator_agent",
    description=(
        "Top-level workflow graph for the AI Research Assistant: coordinates "
        "research, summarization, fact verification, human approval, citation "
        "generation, and final report export as a single linear pipeline."
    ),
    sub_agents=[
        research_agent,
        summarization_agent,
        fact_verification_agent,
        human_approval_agent,
        citation_agent,
        report_generator_agent,
    ],
    before_agent_callback=[initialize_workflow_state, *BEFORE_AGENT_HOOKS],
    after_agent_callback=AFTER_AGENT_HOOKS,
)


# ADK's CLI (`adk run` / `adk web`) and any `Runner` look for a module-level
# `root_agent` by convention -- this is the single entry point into the
# entire multi-agent system.
root_agent = orchestrator_agent
