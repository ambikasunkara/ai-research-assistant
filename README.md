# AI Research Assistant

**A production-shaped, multi-agent research automation system built on Google
Agent Development Kit (ADK) 2.x, submitted for the Kaggle AI Agents Capstone
(Freestyle track).**

Give it a research topic. It searches academic papers (arXiv) and the general
web, synthesizes a research brief, summarizes it, fact-checks the load-bearing
claims against real sources, **pauses to ask a human reviewer for explicit
sign-off**, generates a formatted bibliography, and compiles + exports a final
Markdown report — all while every agent, tool call, and security decision is
audit-logged.

```
Research query
      │
      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        orchestrator_agent (SequentialAgent)             │
│                                                                         │
│  1. research_agent ────────► paper_search_agent  (AgentTool)          │
│        │                          │                                    │
│        │                          └──► MCP: search_papers (arXiv)      │
│        └──► MCP: search_web (DuckDuckGo)                               │
│        │                                                                │
│        ▼  ctx.state["research_findings"]                               │
│  2. summarization_agent                                                │
│        ▼  ctx.state["summary"]                                         │
│  3. fact_verification_agent ──► MCP: fact_check, compare_sources       │
│        ▼  ctx.state["verification_report"]                             │
│  4. human_approval_agent ─────► ADK request_input (HUMAN-IN-THE-LOOP)  │
│        ▼  ctx.state["human_approval_decision"]                         │
│  5. citation_agent ───────────► MCP: generate_citations                │
│        ▼  ctx.state["citations"]                                       │
│  6. report_generator_agent ───► MCP: export_report                    │
│        ▼  ctx.state["final_report"]                                    │
└─────────────────────────────────────────────────────────────────────────┘
      │
      ▼
Markdown report on disk + full JSONL audit trail
```

Every agent above is wrapped in the same **Security Checkpoint** (prompt-
injection screening, PII redaction, rate limiting, tool-argument validation)
and **structured audit logging**, wired in uniformly as ADK lifecycle
callbacks rather than re-implemented per agent.

---

## Table of contents

- [Capstone requirement mapping](#capstone-requirement-mapping)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Running it](#running-it)
- [API reference](#api-reference)
- [Security features](#security-features)
- [Human-in-the-loop approval flow](#human-in-the-loop-approval-flow)
- [Testing](#testing)
- [Docker / deployment](#docker--deployment)
- [Design notes & tradeoffs](#design-notes--tradeoffs)
- [Roadmap](#roadmap)

---

## Capstone requirement mapping

| Requirement | Where it lives | Notes |
|---|---|---|
| **Google ADK 2.x multi-agent system** | `src/ai_research_assistant/agent.py` | 7 `LlmAgent`s composed into a `SequentialAgent` workflow graph (`orchestrator_agent` = `root_agent`). Tested against real `google-adk==2.3.0`. |
| **MCP Server** | `src/ai_research_assistant/mcp_server.py` | Standalone `FastMCP` server exposing 6 tools (`search_papers`, `search_web`, `compare_sources`, `fact_check`, `generate_citations`, `export_report`). Runs over stdio (default, spawned by ADK) or `streamable-http`/`sse` for networked deployment. |
| **AgentTool** | `agent.py` → `research_agent` | `paper_search_agent` is wrapped in `AgentTool(agent=paper_search_agent)` and exposed to `research_agent` as a callable tool, alongside a direct MCP tool. |
| **`ctx.state`** | `agent.py` → `initialize_workflow_state`, every `output_key` | The orchestrator seeds `workflow_id` / `research_query` / `started_at` into `ctx.state` in a `before_agent_callback`; every downstream agent reads/writes state exclusively via `output_key` + `{state_key}` instruction interpolation — there is no out-of-band data passing. |
| **Human-in-the-loop** | `agent.py` → `human_approval_agent`; `fast_api_app.py` → `/approve` | Uses ADK's built-in `request_input` long-running tool to pause the workflow graph and ask a human reviewer to approve/reject before the citation + report stages run. The FastAPI layer exposes `/approve` to resume the paused run. |
| **Security features** | `security.py`, `fast_api_app.py` (`verify_api_key`) | Prompt-injection pattern screening, PII redaction, per-session rate limiting, tool-argument validation (path-traversal guard on `export_report`), and API-key auth at the HTTP boundary. All wired in as ADK callbacks — no agent has to remember to call them. |
| **Deployability** | `Dockerfile`, `docker-compose.yml`, `Makefile` | Multi-stage, non-root Docker image with a healthcheck; `docker compose up` for one-command local deployment; `Makefile` targets for venv/dev workflows. |
| **Production-ready documentation** | This file, `SUBMISSION_WRITEUP.md`, `DEMO_SCRIPT.md`, inline module docstrings | Every module opens with an architecture-level docstring explaining *why*, not just *what*. |

---

## Project structure

```
.
├── src/ai_research_assistant/
│   ├── __init__.py            # exposes `root_agent` for `adk run` / `adk web`
│   ├── agent.py                # the multi-agent workflow graph (the heart of the project)
│   ├── mcp_server.py            # standalone MCP tool server (FastMCP, 6 tools)
│   ├── security.py              # Security Checkpoint (guardrails, callbacks)
│   ├── audit.py                 # structured JSONL audit logging (callbacks)
│   ├── config.py                # typed, centralized settings (pydantic-settings)
│   └── fast_api_app.py          # FastAPI wrapper around the ADK Runner (HTTP API)
├── tests/                       # pytest suite (49 tests, no network/LLM calls required)
├── reports/                     # exported research reports land here (gitignored contents)
├── logs/                        # audit.log.jsonl lands here (gitignored contents)
├── Dockerfile                   # multi-stage, non-root production image
├── docker-compose.yml            # one-command local/deployment orchestration
├── Makefile                      # venv / run / test / lint / docker convenience targets
├── pyproject.toml                 # dependencies, console scripts, tool config
├── .env.example                   # every environment variable, documented
├── README.md                      # you are here
├── SUBMISSION_WRITEUP.md           # Kaggle capstone writeup
├── DEMO_SCRIPT.md                  # step-by-step demo walkthrough
└── LICENSE                         # Apache-2.0
```

---

## Quick start

Requirements: Python 3.11+, a Google Gemini API key (or a GCP project with
Vertex AI enabled).

```bash
git clone <this-repo>
cd ai-research-assistant

make install         # creates .venv and installs the project (pip install -e .)
make env             # copies .env.example -> .env
# now edit .env and set GOOGLE_API_KEY=<your-key>

make run-api         # starts the FastAPI server on http://localhost:8080
```

In another terminal:

```bash
curl -s http://localhost:8080/health

curl -s -X POST http://localhost:8080/research \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me-dev-key" \
  -d '{"query": "Recent advances in retrieval-augmented generation for LLMs", "user_id": "demo-user"}'
```

The response will come back with `"status": "pending_approval"` once the
workflow reaches the Human Approval gate — see
[Human-in-the-loop approval flow](#human-in-the-loop-approval-flow) below for
how to approve it.

You can also drive the agent graph directly through ADK's own CLI/web UI
instead of the FastAPI layer:

```bash
make run-adk-cli     # adk run src/ai_research_assistant
make run-adk-web     # adk web src   (browser UI on http://localhost:8000)
```

---

## Configuration

All configuration is centralized in `config.py` and sourced from environment
variables / a `.env` file (see `.env.example` for the full, documented list).
Nothing else in the codebase reads `os.environ` directly.

Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini Developer API key (or set `GOOGLE_GENAI_USE_VERTEXAI=1` + `GOOGLE_CLOUD_PROJECT` to use Vertex AI instead). |
| `MODEL_PRO` / `MODEL_FLASH` | `gemini-2.5-pro` / `gemini-2.5-flash` | Heavier reasoning agents (orchestrator-adjacent, fact-checking) use `MODEL_PRO`; lighter/high-throughput agents (search, summarization, citations) use `MODEL_FLASH`. |
| `MCP_TRANSPORT` | `stdio` | `stdio` launches the MCP server as a local subprocess (default, simplest). `streamable-http` / `sse` run it as a standalone network service — see [Design notes](#design-notes--tradeoffs). |
| `RESEARCH_API_KEYS` | `change-me-dev-key` | Comma-separated list of valid `X-API-Key` values for the FastAPI layer. **Change this before deploying.** |
| `RATE_LIMIT_PER_MINUTE` | `20` | Per-session sliding-window rate limit enforced by the Security Checkpoint. |
| `BLOCK_PROMPT_INJECTION` | `1` | Toggle for the prompt-injection pattern screen. |
| `REQUIRE_HUMAN_APPROVAL` | `1` | Documents the intended posture; the approval gate itself is always present in the graph (see Roadmap for making this runtime-configurable). |
| `REPORTS_OUTPUT_DIR` / `AUDIT_LOG_DIR` | `./reports` / `./logs` | Where exported reports and the JSONL audit trail are written. |

---

## Running it

### Locally (venv)

```bash
make install-dev     # runtime + dev deps (pytest, ruff, mypy, black)
make env
make run-api         # http://localhost:8080
```

### Standalone MCP server (for inspection/debugging)

```bash
make run-mcp          # runs mcp_server.py directly over stdio
```

You can point any MCP-compatible client (e.g. the `mcp` CLI inspector) at
this to browse the 6 tools independently of the agent graph.

### Docker

```bash
cp .env.example .env    # edit GOOGLE_API_KEY
docker compose up --build
```

This builds the multi-stage image (see `Dockerfile`), runs the FastAPI app on
port 8080, and bind-mounts `./reports` and `./logs` so output survives
container restarts.

---

## API reference

All endpoints except `/health` require an `X-API-Key` header.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check. |
| `POST` | `/research` | Starts (or continues) a research workflow run. Body: `{"query": str, "user_id"?: str, "session_id"?: str}`. |
| `POST` | `/approve` | Resumes a workflow paused at the Human Approval gate. Body: `{"session_id": str, "approved": bool, "comments"?: str}`. |
| `GET` | `/reports/{filename}` | Fetches a previously exported report file. |

`POST /research` and `POST /approve` both return:

```json
{
  "session_id": "session-demo-user-...",
  "status": "completed | pending_approval | error",
  "message": "...",
  "state": { "...ctx.state snapshot at this point..." }
}
```

---

## Security features

Implemented in `security.py` and wired in as ADK lifecycle callbacks so they
apply uniformly to every agent in the graph, plus at the FastAPI boundary:

1. **API-key authentication** (`fast_api_app.py`) — every HTTP request except
   `/health` must present a valid `X-API-Key`.
2. **Prompt-injection / jailbreak screening** (`before_agent_callback`) —
   pattern-matches the incoming user message against a conservative library
   of known injection phrasings ("ignore previous instructions", "developer
   mode", etc.) before it ever reaches a model.
3. **Per-session rate limiting** (`before_agent_callback`) — a sliding-window
   limiter protects the MCP tool backends (arXiv, DuckDuckGo) from abuse.
4. **PII redaction** (`before_model_callback`) — strips emails, phone
   numbers, SSNs, and credit-card-shaped numbers out of outgoing model
   requests as a last line of defense.
5. **Tool-argument validation** (`before_tool_callback`) — validates
   arguments to sensitive tools (in particular `export_report`, which writes
   to disk) against path traversal, and flags abnormally large arguments to
   any tool.
6. **Structured audit logging** (`audit.py`) — every agent start/end, tool
   call start/end, and model call is written as a JSON line to
   `logs/audit.log.jsonl`, including elapsed time, a stable `session_id`, and
   truncated argument/response previews (to avoid bloating the log with e.g.
   full paper abstracts).
7. **Defense-in-depth path safety** — both `export_report` (in the MCP
   server) and `GET /reports/{filename}` (in the API layer) independently
   resolve and re-validate that the final path stays inside the configured
   reports directory, even though the filename is already sanitized upstream.

---

## Human-in-the-loop approval flow

The `human_approval_agent` calls ADK's built-in `request_input` long-running
tool with a recap of the summary + verification confidence. When this
happens:

1. `runner.run_async()` yields an `Event` whose `long_running_tool_ids`
   contains the pending function call's id.
2. `fast_api_app.py` detects this, records the pending approval (keyed by
   `session_id`), and returns `status: "pending_approval"` with the
   human-readable prompt in `message`.
3. A human reviewer calls `POST /approve` with `{"session_id", "approved":
   true|false, "comments"?}`.
4. The API builds a `FunctionResponse` echoing the same function-call id and
   the literal name ADK's `request_input` tool is registered under
   (`adk_request_input`), and resumes the exact same session.
5. If approved, `citation_agent` and `report_generator_agent` run and a
   Markdown report is exported. If rejected, `report_generator_agent`
   explicitly skips the export tool call and explains that the human
   reviewer declined.

Example:

```bash
# 1. Kick off a research run
curl -s -X POST http://localhost:8080/research \
  -H "X-API-Key: change-me-dev-key" -H "Content-Type: application/json" \
  -d '{"query": "Mixture-of-experts architectures for LLMs", "user_id": "alice", "session_id": "demo-1"}'
# -> {"session_id": "demo-1", "status": "pending_approval", "message": "...(recap + 'Approve publishing the final report? (yes/no)')..."}

# 2. Approve it
curl -s -X POST http://localhost:8080/approve \
  -H "X-API-Key: change-me-dev-key" -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "approved": true, "comments": "Looks solid, publish it."}'
# -> {"status": "completed", "message": "...", "state": {"final_report": "..."}}

# 3. Fetch the exported report
curl -s -H "X-API-Key: change-me-dev-key" http://localhost:8080/reports/<filename-from-final_report>.md
```

---

## Testing

```bash
make test
# or directly:
PYTHONPATH=src pytest -v
```

49 tests, organized as:

- `tests/test_mcp_tools.py` — the deterministic MCP tools (`compare_sources`,
  `fact_check`, `generate_citations`, `export_report`) tested directly;
  `search_papers` / `search_web` tested with `httpx` mocked out (no real
  network calls in the unit test suite).
- `tests/test_security.py` — every Security Checkpoint guardrail
  (injection screening, PII redaction, rate limiting, tool-arg validation).
- `tests/test_audit.py` — JSONL audit record shape, append behavior, and
  graceful failure handling.
- `tests/test_agent_graph.py` — structural smoke tests on the ADK workflow
  graph: sub-agent order, the `ctx.state` output-key contract between
  stages, `AgentTool`/MCP wiring, and that every agent has security + audit
  callbacks attached. **No LLM calls, no API costs.**
- `tests/test_api.py` — the FastAPI HTTP surface (auth enforcement, health
  check, report path-traversal protection) via `TestClient`.

None of the tests require a real `GOOGLE_API_KEY`, a running MCP server
subprocess, or network access — `tests/conftest.py` sets safe dummy
environment variables before any project module is imported.

---

## Docker / deployment

`Dockerfile` is a multi-stage build:

- **Builder stage** installs the package (and its dependencies) into
  `/install` using `pip install --prefix=/install .`.
- **Runtime stage** copies only the installed package + source into a slim
  `python:3.11-slim` image, runs as a non-root `appuser`, and exposes a
  `HEALTHCHECK` against `/health`.
- The container's only public interface is the FastAPI app; it internally
  spawns the MCP server as a stdio subprocess per `McpToolset` (see
  `agent.py`), so no separate MCP container is required for this deployment
  mode.

```bash
make docker-build
make docker-run       # or: docker compose up --build
```

For a real production deployment, two `InMemory*` components in
`fast_api_app.py` should be swapped out (see inline comments at their
definition sites):

- `InMemorySessionService` → `DatabaseSessionService` or
  `VertexAiSessionService`, so sessions (and pending human approvals)
  survive process restarts and work across multiple API workers/replicas.
- `_pending_approvals` (an in-process dict) → persisted alongside session
  state in the same backing store.

---

## Design notes & tradeoffs

- **`SequentialAgent` vs. the newer `Workflow` graph API.** As of
  `google-adk` 2.3.x, `SequentialAgent` is marked deprecated in favor of a
  newer `Workflow`/graph-node API, but remains fully functional. This
  project deliberately targets the classic, extensively-documented
  `LlmAgent` + `SequentialAgent` + lifecycle-callback model for the
  capstone submission, since it is the most broadly supported surface at
  time of writing. See [Roadmap](#roadmap).
- **`stdio` vs. `streamable-http` MCP transport.** Each `McpToolset` in
  `agent.py` currently launches its own `stdio` subprocess of
  `mcp_server.py` (one per tool-filtered toolset, for least-privilege tool
  scoping). This is simple and has no extra moving parts to deploy, but it
  means the MCP server process count scales with the number of distinct
  tool-filter groups. Setting `MCP_TRANSPORT=streamable-http` and running
  `mcp_server.py` as one standalone service — then pointing every
  `McpToolset` at the same `StreamableHTTPConnectionParams` URL — collapses
  this to a single shared server process, at the cost of one more service
  to deploy and secure.
- **Deterministic-heuristic + LLM-judgment fact-checking.** `fact_check` and
  `compare_sources` in `mcp_server.py` are intentionally simple, dependency-
  free lexical-overlap heuristics rather than a second LLM call. They exist
  to give `fact_verification_agent` concrete, inspectable *evidence*
  (which sources actually share vocabulary with a claim) to reason over,
  rather than asking the model to fact-check purely from parametric memory.
  The confidence judgment itself is still made by the LLM.
- **Pattern-matching prompt-injection screening.** `security.py`'s injection
  screen is a conservative regex library, not a classifier model. It is a
  fast, dependency-free first line of defense, explicitly documented as
  such — a hardened production deployment would likely layer a dedicated
  classifier or guardrails service on top.

---

## Roadmap

- Migrate `orchestrator_agent` from `SequentialAgent` to ADK's newer
  `Workflow` graph API once it stabilizes further, to stay ahead of the
  deprecation.
- Make `REQUIRE_HUMAN_APPROVAL` actually branch the graph at runtime
  (skip the approval gate entirely in fully-automated/batch deployments)
  rather than only documenting the intended posture.
- Swap `InMemorySessionService` for `DatabaseSessionService` and persist
  `_pending_approvals` alongside it for multi-worker, restart-safe
  deployments.
- Add a lightweight web UI for the human-approval step (currently a raw
  `curl`/HTTP call) so a non-technical reviewer can approve/reject reports.
- Layer a dedicated prompt-injection classifier alongside the current
  pattern-matching screen.

---

## License

Apache-2.0 — see [LICENSE](LICENSE).
# ai-research-assistant
