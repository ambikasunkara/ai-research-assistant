"""
conftest.py
===========
Shared pytest fixtures. Sets safe, deterministic environment variables
*before* `ai_research_assistant.config` is ever imported, so the test suite
never touches a real Google API key, never rate-limits itself out of its own
tests, and writes reports/logs to a throwaway temp directory instead of the
real project `reports/` / `logs/` folders.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# --------------------------------------------------------------------------
# Environment must be set BEFORE any `ai_research_assistant.*` module is
# imported anywhere in the test session, since `config.settings` is built
# once at import time (`get_settings()` is `lru_cache`d).
# --------------------------------------------------------------------------
_TMP_DIR = tempfile.mkdtemp(prefix="ai-research-assistant-tests-")

os.environ.setdefault("GOOGLE_API_KEY", "test-key-for-unit-tests")
os.environ.setdefault("RESEARCH_API_KEYS", "test-api-key")
os.environ.setdefault("REPORTS_OUTPUT_DIR", os.path.join(_TMP_DIR, "reports"))
os.environ.setdefault("AUDIT_LOG_DIR", os.path.join(_TMP_DIR, "logs"))
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000")
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture()
def api_key() -> str:
    """The API key the test suite's own FastAPI settings expect."""
    from ai_research_assistant.config import settings

    return next(iter(settings.research_api_keys_set))
