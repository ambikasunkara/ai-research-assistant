"""
AI Research Assistant
======================
A multi-agent research automation system built on Google ADK 2.x.

Exposes `root_agent`, the top-level orchestrator, so that `adk run`,
`adk web`, and any custom `Runner` can discover the agent graph via
`from ai_research_assistant import root_agent`.
"""

from ai_research_assistant.agent import root_agent

__all__ = ["root_agent"]

__version__ = "1.0.0"
