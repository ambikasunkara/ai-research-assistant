"""Unit tests for audit.py -- structured JSONL audit logging."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ai_research_assistant.audit import AuditLogger


def test_log_event_writes_one_json_line_with_expected_fields():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "audit.log.jsonl"
        logger = AuditLogger(log_path=str(log_path))

        logger.log_event(
            "TOOL_CALL_START",
            session_id="sess-123",
            invocation_id="inv-456",
            agent_name="research_agent",
            tool_name="search_papers",
            details={"query": "transformers"},
            severity="INFO",
        )

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["event_type"] == "TOOL_CALL_START"
        assert record["session_id"] == "sess-123"
        assert record["invocation_id"] == "inv-456"
        assert record["agent_name"] == "research_agent"
        assert record["tool_name"] == "search_papers"
        assert record["severity"] == "INFO"
        assert record["details"] == {"query": "transformers"}
        assert "id" in record and "timestamp" in record


def test_log_event_appends_multiple_records():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "audit.log.jsonl"
        logger = AuditLogger(log_path=str(log_path))

        for i in range(3):
            logger.log_event("AGENT_START", agent_name=f"agent_{i}")

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3


def test_log_event_never_raises_when_log_dir_is_unwritable(tmp_path, monkeypatch):
    # Point at a path inside a directory that doesn't exist and can't be
    # created (a file where a directory is expected) to force an OSError,
    # and confirm log_event swallows it rather than propagating.
    bogus_file = tmp_path / "not_a_directory"
    bogus_file.write_text("x")
    unwritable_path = bogus_file / "audit.log.jsonl"

    logger = AuditLogger(log_path=str(unwritable_path))
    # Should not raise, even though the path is invalid.
    logger.log_event("AGENT_START", agent_name="whatever")


def test_log_event_truncates_oversized_details_via_safe_truncate():
    from ai_research_assistant.audit import _safe_truncate

    huge_payload = {"data": "x" * 10_000}
    truncated = _safe_truncate(huge_payload, limit=100)
    assert isinstance(truncated, str)
    assert len(truncated) <= 100 + len("...<truncated>")
    assert truncated.endswith("...<truncated>")


def test_safe_truncate_passes_through_small_payloads_unchanged():
    from ai_research_assistant.audit import _safe_truncate

    small_payload = {"a": 1}
    assert _safe_truncate(small_payload, limit=800) == small_payload
