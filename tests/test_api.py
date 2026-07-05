"""
API-layer tests for fast_api_app.py.

These tests exercise the HTTP surface (health check, API-key enforcement,
report path-traversal protection) without ever invoking the LLM or the ADK
`Runner` pipeline -- that is covered structurally by test_agent_graph.py and
would otherwise require real Gemini credentials / network access.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from ai_research_assistant.fast_api_app import app
from ai_research_assistant.config import settings

client = TestClient(app)


def test_health_check_does_not_require_auth():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_research_endpoint_rejects_missing_api_key():
    response = client.post("/research", json={"query": "What is retrieval-augmented generation?"})
    assert response.status_code == 401


def test_research_endpoint_rejects_wrong_api_key():
    response = client.post(
        "/research",
        json={"query": "What is retrieval-augmented generation?"},
        headers={"X-API-Key": "definitely-not-the-right-key"},
    )
    assert response.status_code == 401


def test_approve_endpoint_requires_auth():
    response = client.post(
        "/approve",
        json={"session_id": "nonexistent", "approved": True},
    )
    assert response.status_code == 401


def test_approve_endpoint_404s_for_unknown_session(api_key):
    response = client.post(
        "/approve",
        json={"session_id": "does-not-exist", "approved": True},
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 404


def test_get_report_requires_auth():
    response = client.get("/reports/whatever.md")
    assert response.status_code == 401


def test_get_report_blocks_path_traversal(api_key):
    # Path.name strips any directory components, so this resolves to a
    # (non-existent) file literally named "passwd" inside the reports dir --
    # never anything outside it.
    response = client.get(
        "/reports/..%2F..%2F..%2Fetc%2Fpasswd",
        headers={"X-API-Key": api_key},
    )
    assert response.status_code in (404, 400)


def test_get_report_404s_for_missing_file(api_key):
    response = client.get("/reports/does-not-exist.md", headers={"X-API-Key": api_key})
    assert response.status_code == 404


def test_get_report_serves_existing_file(api_key):
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.reports_dir / "sample-report.md"
    report_path.write_text("# Sample Report\n\nHello.", encoding="utf-8")

    response = client.get("/reports/sample-report.md", headers={"X-API-Key": api_key})
    assert response.status_code == 200
    assert b"Sample Report" in response.content
