# Kaggle AI Agents Capstone — Submission Writeup

**Project:** AI Research Assistant
**Track:** Freestyle
**Framework:** Google Agent Development Kit (ADK) 2.x, tested against
`google-adk==2.3.0`

---

## 1. Problem statement

Doing a literature review or background-research pass on a topic is
repetitive and error-prone in the same few ways every time: searching
multiple sources, synthesizing overlapping/contradictory findings,
double-checking that a summary's key claims are actually supported by the
sources it cites, and formatting a bibliography correctly. It is also a
task where **fully automating away the human** is the wrong goal — a
researcher should stay in the loop before a report goes out the door,
especially since LLM-written summaries can plausibly misstate a source.

**Goal:** build a multi-agent system that does the repetitive 80% (search,
synthesize, verify, cite, format) end-to-end, but pauses for explicit human
sign-off before anything is finalized — with the security and audit posture
of something you'd actually be willing to run against real traffic, not
just a notebook demo.

---

## 2. Approach

The system is a single ADK **workflow graph**: a `SequentialAgent`
(`orchestrator_agent`, exposed as `root_agent`) chaining six `LlmAgent`
stages, plus one specialist sub-agent invoked as an `AgentTool`:

```
research_agent → summarization_agent → fact_verification_agent
    → human_approval_agent → citation_agent → report_generator_agent
```

Each stage:

- reads what it needs from **`ctx.state`** via `{state_key}` instruction
  interpolation (ADK resolves this automatically before every model call),
  and
- writes its own result back into `ctx.state` via `output_key`,

so the entire pipeline's data flow is `ctx.state` — there is no hidden
side-channel between agents. `initialize_workflow_state` (a
`before_agent_callback` on the orchestrator itself) seeds `workflow_id`,
`research_query`, and `started_at` into state before any sub-agent runs, so
every downstream agent and every audit-log line can reference a stable
identity for the run.

**Tool access is provided almost entirely through a standalone MCP server**
(`mcp_server.py`, built on `FastMCP`) exposing six tools: `search_papers`
(arXiv), `search_web` (DuckDuckGo), `compare_sources`, `fact_check`,
`generate_citations`, and `export_report`. Rather than giving every agent
the full tool surface, each agent gets its own `McpToolset` instance with a
`tool_filter` scoped to exactly the tools it needs — principle of least
privilege applied at the tool level, not just documented as an intention.

**The research step uses `AgentTool`:** `research_agent` delegates academic
search to a narrow, single-purpose `paper_search_agent` by wrapping it in
`AgentTool(agent=paper_search_agent)` and calling it like any other tool,
while also calling the `search_web` MCP tool directly for non-academic
context. This demonstrates agent-as-tool composition alongside plain MCP
tool use in the same agent.

**Human-in-the-loop is a first-class pipeline stage, not a bolt-on:**
`human_approval_agent` calls ADK's built-in `request_input` long-running
tool, which pauses the entire workflow graph mid-run. The FastAPI layer
(`fast_api_app.py`) detects the pause via `event.long_running_tool_ids`,
returns a `pending_approval` status with a human-readable recap, and
resumes the *exact same session* once `POST /approve` supplies the human's
decision via a `FunctionResponse`. If rejected, the terminal
`report_generator_agent` explicitly skips the file-export tool call — the
approval gate has real teeth, not just a logged opinion.

**Security and audit are cross-cutting, not per-agent.** Rather than
threading guardrail calls through every agent's instruction or tool
implementation, `security.py` and `audit.py` are wired in once as ADK
lifecycle callbacks (`before_agent_callback`, `before_model_callback`,
`before_tool_callback`, `after_tool_callback`, `after_agent_callback`) and
attached to every agent identically. This means: (a) no agent can
accidentally skip a guardrail, and (b) adding a new agent to the graph
automatically inherits the full security/audit posture.

---

## 3. Architecture

See the diagram and full requirement-to-code mapping table in
[`README.md`](README.md#capstone-requirement-mapping). In short:

| Layer | Component |
|---|---|
| Multi-agent orchestration | `agent.py` — 7 `LlmAgent`s + 1 `SequentialAgent` |
| Tool server | `mcp_server.py` — `FastMCP`, 6 tools, stdio/HTTP/SSE transports |
| Guardrails | `security.py` — injection screening, PII redaction, rate limiting, tool-arg validation |
| Observability | `audit.py` — structured JSONL audit trail |
| Config | `config.py` — typed settings via `pydantic-settings`, single source of truth |
| HTTP API | `fast_api_app.py` — FastAPI wrapper around the ADK `Runner`, API-key auth, human-approval resumption |
| Deployment | `Dockerfile`, `docker-compose.yml`, `Makefile` |

---

## 4. What makes this "production-shaped" rather than a notebook demo

1. **Typed, centralized configuration** (`config.py`) — every environment
   variable is validated and typed via `pydantic-settings`; no module reads
   `os.environ` directly.
2. **Defense-in-depth path safety.** Both `export_report` (MCP server) and
   `GET /reports/{filename}` (API layer) independently re-resolve and
   verify that the final file path is contained within the configured
   reports directory, even though the filename is already sanitized
   upstream in each case — a second layer catches a bug in the first.
3. **A real audit trail**, not print statements: every agent
   start/end, tool call start/end, and model call is a structured JSON
   line with a stable `session_id`/`invocation_id`, written to
   `logs/audit.log.jsonl`, ready to ship to a SIEM.
4. **Non-root, multi-stage, health-checked Docker image.**
5. **A 49-test pytest suite** covering the guardrails, the audit logger,
   the MCP tools (including mocked-network tests for the two network-
   calling tools), the HTTP API surface, and — critically — the *structure*
   of the agent graph itself (sub-agent order, the `ctx.state` output-key
   contract between stages, tool wiring, and that every single agent has
   the security/audit callbacks attached). None of this requires a real
   `GOOGLE_API_KEY` or network access to run.
6. **Everything validated against the real, installed `google-adk==2.3.0`
   package** during development of this submission — not written from
   memory against an assumed API surface. Import paths, callback
   signatures (`Context`/`CallbackContext` unification, `before_tool_callback`
   argument order, etc.), `McpToolset`/`StdioConnectionParams` constructor
   signatures, `Runner.run_async` keyword arguments, and the
   `request_input` long-running-tool function name
   (`adk_request_input`) were all checked against the installed package
   source, and two real bugs (an SSN/phone regex ordering bug in PII
   redaction, and a citation-formatting bug for papers with no listed
   authors) were caught by the test suite and fixed.

---

## 5. Known limitations & honest tradeoffs

- **`SequentialAgent` is deprecated (but functional) in `google-adk`
  2.3.x**, in favor of a newer `Workflow` graph API. This submission
  targets the classic, more broadly documented `LlmAgent` +
  `SequentialAgent` + callback model deliberately, and calls out the
  migration path in `README.md`'s Roadmap section.
- **In-memory session/approval state.** `InMemorySessionService` and the
  `_pending_approvals` dict in `fast_api_app.py` are fine for a local demo
  or single-process deployment, but do not survive a restart or scale
  across multiple API workers. The README documents the swap to
  `DatabaseSessionService`/`VertexAiSessionService` for a real deployment.
- **Prompt-injection screening is pattern-based, not a classifier.** It is
  explicitly documented as a fast, dependency-free first line of defense,
  not a complete solution.
- **`fact_check`/`compare_sources` are deterministic lexical-overlap
  heuristics**, not a second LLM judgment — they exist to give the
  fact-verification LLM concrete evidence to reason over, not to replace
  its judgment.

---

## 6. How to run it

See [`README.md`](README.md#quick-start) for full setup, and
[`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) for a step-by-step walkthrough including
exact `curl` commands and expected output at each stage.
