"""Unit tests for the deterministic MCP tools in mcp_server.py.

`compare_sources`, `fact_check`, `generate_citations`, and `export_report`
are pure/deterministic (no network calls), so they are tested directly.
`search_papers` / `search_web` are tested with `httpx` mocked out, since
real network access should never be a dependency of the unit test suite.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ai_research_assistant import mcp_server


# --------------------------------------------------------------------------
# compare_sources
# --------------------------------------------------------------------------
def test_compare_sources_computes_jaccard_overlap():
    sources = [
        {"title": "Paper A", "summary": "Attention mechanisms improve transformer performance."},
        {"title": "Paper B", "summary": "Transformer models rely on attention mechanisms."},
    ]
    result = mcp_server.compare_sources(sources)

    assert len(result["coverage_summary"]) == 2
    assert len(result["comparison_matrix"]) == 1
    pair = result["comparison_matrix"][0]
    assert pair["source_a"] == "Paper A"
    assert pair["source_b"] == "Paper B"
    assert 0.0 < pair["similarity_score"] <= 1.0
    assert "attention" in pair["shared_keywords"]


def test_compare_sources_handles_empty_list():
    result = mcp_server.compare_sources([])
    assert result == {"coverage_summary": [], "comparison_matrix": []}


def test_compare_sources_handles_single_source_no_pairs():
    result = mcp_server.compare_sources([{"title": "Only One", "summary": "Some text here."}])
    assert len(result["coverage_summary"]) == 1
    assert result["comparison_matrix"] == []


# --------------------------------------------------------------------------
# fact_check
# --------------------------------------------------------------------------
def test_fact_check_supported_claim():
    sources = [
        {
            "title": "Relevant Paper",
            "summary": "Attention mechanisms improve transformer performance significantly.",
        }
    ]
    result = mcp_server.fact_check("Attention mechanisms improve transformer performance", sources)
    assert result["unsupported"] is False
    assert result["confidence_score"] > 0
    assert result["supporting_sources"][0]["title"] == "Relevant Paper"


def test_fact_check_unsupported_claim():
    sources = [{"title": "Unrelated Paper", "summary": "This is about cooking pasta recipes."}]
    result = mcp_server.fact_check("Quantum computers use superconducting qubits", sources)
    assert result["unsupported"] is True
    assert result["confidence_score"] == 0.0


def test_fact_check_empty_claim_returns_zero_confidence():
    result = mcp_server.fact_check("   ", [{"title": "X", "summary": "y"}])
    assert result["confidence_score"] == 0.0
    assert result["unsupported"] is True


# --------------------------------------------------------------------------
# generate_citations
# --------------------------------------------------------------------------
@pytest.mark.parametrize("style", ["APA", "MLA", "Chicago"])
def test_generate_citations_all_styles(style):
    papers = [
        {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "published": "2017-06-12",
            "url": "https://arxiv.org/abs/1706.03762",
        }
    ]
    result = mcp_server.generate_citations(papers, style=style)
    assert result["style"] == style
    assert result["count"] == 1
    assert "Attention Is All You Need" in result["citations"][0]


def test_generate_citations_handles_missing_fields_gracefully():
    result = mcp_server.generate_citations([{"title": "Untitled Work"}], style="APA")
    assert result["count"] == 1
    assert "Unknown Author" in result["citations"][0]


# --------------------------------------------------------------------------
# export_report
# --------------------------------------------------------------------------
def test_export_report_writes_markdown_file():
    result = mcp_server.export_report("My Test Report", "Report body content.", format="markdown")
    assert "error" not in result
    assert result["file_path"].endswith(".md")
    with open(result["file_path"], encoding="utf-8") as f:
        content = f.read()
    assert "My Test Report" in content
    assert "Report body content." in content


def test_export_report_rejects_path_traversal_via_sanitization():
    # Even a maximally hostile title must resolve to a safe filename inside
    # the configured reports directory -- the slug sanitizer strips '/' etc.
    result = mcp_server.export_report("../../../etc/passwd", "malicious", format="text")
    assert "error" not in result
    from ai_research_assistant.config import settings

    assert settings.reports_dir.resolve() in __import__("pathlib").Path(result["file_path"]).resolve().parents


def test_export_report_html_escapes_content():
    result = mcp_server.export_report("XSS<script>", "<script>alert(1)</script>", format="html")
    with open(result["file_path"], encoding="utf-8") as f:
        content = f.read()
    assert "<script>alert(1)</script>" not in content
    assert "&lt;script&gt;" in content


# --------------------------------------------------------------------------
# search_papers / search_web (network mocked)
# --------------------------------------------------------------------------
_ARXIV_ATOM_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models are based on complex
    recurrent or convolutional neural networks.</summary>
    <published>2017-06-12T17:57:34Z</published>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
  </entry>
</feed>
"""


@pytest.mark.asyncio
async def test_search_papers_parses_arxiv_response(monkeypatch):
    class _FakeResponse:
        text = _ARXIV_ATOM_SAMPLE

        def raise_for_status(self):
            return None

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    result = await mcp_server.search_papers("attention transformers", max_results=5)
    assert result["source"] == "arxiv"
    assert len(result["papers"]) == 1
    assert result["papers"][0]["title"] == "Attention Is All You Need"
    assert result["papers"][0]["paper_id"] == "1706.03762v5"


@pytest.mark.asyncio
async def test_search_papers_handles_network_failure_gracefully(monkeypatch):
    class _FailingAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            raise httpx.ConnectError("simulated network failure")

    monkeypatch.setattr(httpx, "AsyncClient", _FailingAsyncClient)

    result = await mcp_server.search_papers("anything", max_results=3)
    assert result["papers"] == []
    assert "error" in result


@pytest.mark.asyncio
async def test_search_web_parses_duckduckgo_response(monkeypatch):
    payload = {
        "Heading": "Transformer (machine learning)",
        "AbstractText": "A transformer is a deep learning architecture.",
        "AbstractURL": "https://en.wikipedia.org/wiki/Transformer_(machine_learning)",
        "RelatedTopics": [
            {"Text": "BERT - A transformer-based model", "FirstURL": "https://example.com/bert"}
        ],
    }

    class _FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    result = await mcp_server.search_web("transformers", max_results=5)
    assert result["source"] == "duckduckgo"
    assert len(result["results"]) == 2
    assert result["results"][0]["title"] == "Transformer (machine learning)"
