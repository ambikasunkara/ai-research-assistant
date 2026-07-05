"""
Structural smoke tests for the ADK multi-agent workflow graph in agent.py.

These tests do NOT call any LLM (no network, no API costs) -- they only
verify that the graph is *constructed* correctly: the right sub-agents in
the right order, the right `output_key`s (the `ctx.state` contract between
agents), the right tools (including the `AgentTool` and MCP tool wiring),
and the right security/audit callbacks attached to every node. This is
exactly the kind of thing that silently breaks during refactors, so it is
worth locking down with tests even though it never touches the network.
"""

from __future__ import annotations

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

from ai_research_assistant import agent as agent_module


def test_root_agent_is_the_orchestrator():
    assert agent_module.root_agent is agent_module.orchestrator_agent


def test_orchestrator_is_a_sequential_agent_with_six_stages_in_order():
    orch = agent_module.orchestrator_agent
    assert isinstance(orch, SequentialAgent)

    expected_order = [
        "research_agent",
        "summarization_agent",
        "fact_verification_agent",
        "human_approval_agent",
        "citation_agent",
        "report_generator_agent",
    ]
    actual_order = [sub.name for sub in orch.sub_agents]
    assert actual_order == expected_order


def test_ctx_state_contract_output_keys_chain_correctly():
    """Every stage's output_key must be exactly what the next stage's
    instruction interpolates via `{state_key}` -- this is the `ctx.state`
    data-flow contract for the whole pipeline."""
    expected_output_keys = {
        "research_agent": "research_findings",
        "summarization_agent": "summary",
        "fact_verification_agent": "verification_report",
        "human_approval_agent": "human_approval_decision",
        "citation_agent": "citations",
        "report_generator_agent": "final_report",
    }
    by_name = {sub.name: sub for sub in agent_module.orchestrator_agent.sub_agents}
    for name, expected_key in expected_output_keys.items():
        assert by_name[name].output_key == expected_key

    # Spot-check that downstream instructions actually reference the
    # upstream state keys they depend on.
    assert "{research_query}" in by_name["research_agent"].instruction
    assert "{research_findings}" in by_name["summarization_agent"].instruction
    assert "{summary}" in by_name["fact_verification_agent"].instruction
    assert "{verification_report}" in by_name["human_approval_agent"].instruction
    assert "{human_approval_decision}" in by_name["report_generator_agent"].instruction
    assert "{citations}" in by_name["report_generator_agent"].instruction


def test_research_agent_uses_agent_tool_and_mcp_tool():
    research_agent = agent_module.research_agent
    tool_types = [type(t) for t in research_agent.tools]
    assert AgentTool in tool_types
    assert McpToolset in tool_types

    agent_tool = next(t for t in research_agent.tools if isinstance(t, AgentTool))
    assert agent_tool.agent is agent_module.paper_search_agent


def test_human_approval_agent_uses_request_input_long_running_tool():
    from google.adk.tools import request_input

    human_approval_agent = agent_module.human_approval_agent
    assert request_input in human_approval_agent.tools


def test_every_llm_agent_in_the_graph_has_security_and_audit_callbacks():
    """Defense-in-depth: every node -- not just some -- must run through the
    Security Checkpoint and the audit trail."""
    all_agents = [
        agent_module.paper_search_agent,
        agent_module.research_agent,
        agent_module.summarization_agent,
        agent_module.fact_verification_agent,
        agent_module.human_approval_agent,
        agent_module.citation_agent,
        agent_module.report_generator_agent,
    ]
    for a in all_agents:
        assert isinstance(a, LlmAgent)
        before_agent = a.before_agent_callback or []
        before_agent = before_agent if isinstance(before_agent, list) else [before_agent]
        assert agent_module.security_before_agent in before_agent
        assert agent_module.audit_before_agent in before_agent

        before_model = a.before_model_callback or []
        before_model = before_model if isinstance(before_model, list) else [before_model]
        assert agent_module.security_before_model in before_model
        assert agent_module.audit_before_model in before_model


def test_orchestrator_initializes_workflow_state_first():
    """`initialize_workflow_state` must run before the security/audit hooks
    on the orchestrator so that `workflow_id` / `research_query` are always
    present in `ctx.state` for every downstream agent and audit record."""
    before_agent = agent_module.orchestrator_agent.before_agent_callback
    assert before_agent[0] is agent_module.initialize_workflow_state


def test_mcp_toolsets_are_scoped_to_least_privilege_per_agent():
    """Each agent's MCP tool_filter should expose only the tools it needs --
    this is the principle-of-least-privilege check for the MCP surface."""
    assert agent_module.paper_search_tools.tool_filter == ["search_papers"]
    assert agent_module.web_search_tools.tool_filter == ["search_web"]
    assert set(agent_module.verification_tools.tool_filter) == {"fact_check", "compare_sources"}
    assert agent_module.citation_tools.tool_filter == ["generate_citations"]
    assert agent_module.report_tools.tool_filter == ["export_report"]
