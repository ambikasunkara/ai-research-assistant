"""
mcp_server.py
=============
A standalone Model Context Protocol (MCP) server that exposes the six
research tools used by the AI Research Assistant:

    * search_papers      - queries the arXiv API for academic papers
    * search_web         - lightweight general web search (DuckDuckGo)
    * compare_sources     - compares multiple sources/papers structurally
    * fact_check          - checks a claim against a set of source documents
    * generate_citations  - formats bibliographic citations (APA/MLA/Chicago)
    * export_report       - writes the final report to disk (Markdown/HTML)

This server is transport-agnostic: it can run over stdio (the default,
launched as a subprocess by ADK's `McpToolset`) or over streamable-HTTP for
a networked deployment. See `main()` at the bottom of this file.

Run directly for local testing:
    python -m ai_research_assistant.mcp_server
"""

from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from xml.etree import ElementTree as ET

import httpx
from mcp.server.fastmcp import FastMCP

from ai_research_assistant.config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("ai_research_assistant.mcp_server")

# --------------------------------------------------------------------------
# FastMCP server instance
# --------------------------------------------------------------------------
mcp = FastMCP(
    name="research-tools",
    instructions=(
        "Tools for academic and web research: searching papers and the web, "
        "comparing sources, fact-checking claims, generating citations, and "
        "exporting a final report to disk."
    ),
    host=settings.mcp_server_host,
    port=settings.mcp_server_port,
)

ARXIV_API_URL = "https://export.arxiv.org/api/query"
DUCKDUCKGO_API_URL = "https://api.duckduckgo.com/"

_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


# --------------------------------------------------------------------------
# Tool 1: search_papers
# --------------------------------------------------------------------------
@mcp.tool()
async def search_papers(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search arXiv for academic papers relevant to a research query.

    Args:
        query: The research topic or keywords to search for.
        max_results: Maximum number of papers to return (capped for safety).

    Returns:
        A dict with a "papers" list. Each paper has: title, authors,
        summary, published, url, and paper_id.
    """
    max_results = max(1, min(max_results, settings.max_papers_per_query))
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(ARXIV_API_URL, params=params)
            resp.raise_for_status()
            papers = _parse_arxiv_feed(resp.text)
    except Exception as exc:
        logger.warning("arXiv search failed (%s); returning empty result set.", exc)
        return {"papers": [], "source": "arxiv", "error": str(exc)}

    return {"papers": papers, "source": "arxiv", "query": query}


def _parse_arxiv_feed(xml_text: str) -> list[dict[str, Any]]:
    """Parses the Atom XML returned by the arXiv API into simple dicts."""
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        title = (entry.findtext("atom:title", default="", namespaces=_ARXIV_NS) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=_ARXIV_NS) or "").strip()
        published = entry.findtext("atom:published", default="", namespaces=_ARXIV_NS) or ""
        paper_id = entry.findtext("atom:id", default="", namespaces=_ARXIV_NS) or ""
        authors = [
            (a.findtext("atom:name", default="", namespaces=_ARXIV_NS) or "").strip()
            for a in entry.findall("atom:author", _ARXIV_NS)
        ]
        if not title:
            continue
        papers.append(
            {
                "paper_id": paper_id.rsplit("/", 1)[-1] if paper_id else "",
                "title": re.sub(r"\s+", " ", title),
                "authors": authors,
                "summary": re.sub(r"\s+", " ", summary),
                "published": published,
                "url": paper_id,
            }
        )
    return papers


# --------------------------------------------------------------------------
# Tool 2: search_web
# --------------------------------------------------------------------------
@mcp.tool()
async def search_web(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the general web for background / non-academic context.

    Uses the DuckDuckGo Instant Answer API (no API key required). This is
    intentionally lightweight -- it supplements, rather than replaces,
    `search_papers` for peer-reviewed sources.

    Args:
        query: The search query.
        max_results: Maximum number of related results to return.

    Returns:
        A dict with an "results" list of {title, snippet, url}.
    """
    params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1, "skip_disambig": 1}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(DUCKDUCKGO_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("Web search failed (%s); returning empty result set.", exc)
        return {"results": [], "source": "duckduckgo", "error": str(exc)}

    results: list[dict[str, str]] = []
    if data.get("AbstractText"):
        results.append(
            {
                "title": data.get("Heading", query),
                "snippet": data["AbstractText"],
                "url": data.get("AbstractURL", ""),
            }
        )
    for topic in data.get("RelatedTopics", []):
        if len(results) >= max_results:
            break
        if isinstance(topic, dict) and topic.get("Text"):
            results.append(
                {
                    "title": topic.get("Text", "").split(" - ")[0][:120],
                    "snippet": html.unescape(topic.get("Text", "")),
                    "url": topic.get("FirstURL", ""),
                }
            )
    return {"results": results[:max_results], "source": "duckduckgo", "query": query}


# --------------------------------------------------------------------------
# Tool 3: compare_sources
# --------------------------------------------------------------------------
@mcp.tool()
def compare_sources(sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare multiple sources (papers or web results) for overlap and coverage.

    Produces a lightweight, deterministic comparison matrix -- keyword
    overlap between sources, relative recency, and relative summary length --
    which the calling agent can use as evidence when synthesizing findings.

    Args:
        sources: A list of source dicts, each expected to have at least a
            "title" and a "summary" (or "snippet") field.

    Returns:
        A dict with "comparison_matrix" (pairwise keyword overlap scores)
        and "coverage_summary" (per-source stats).
    """
    normalized = []
    for src in sources:
        text = (src.get("summary") or src.get("snippet") or "").lower()
        keywords = set(re.findall(r"[a-z]{4,}", text))
        normalized.append(
            {
                "title": src.get("title", "untitled"),
                "keywords": keywords,
                "length": len(text),
                "published": src.get("published", ""),
            }
        )

    coverage_summary = [
        {
            "title": n["title"],
            "summary_length_chars": n["length"],
            "unique_keyword_count": len(n["keywords"]),
            "published": n["published"],
        }
        for n in normalized
    ]

    comparison_matrix = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            a, b = normalized[i], normalized[j]
            union = a["keywords"] | b["keywords"]
            overlap = a["keywords"] & b["keywords"]
            jaccard = len(overlap) / len(union) if union else 0.0
            comparison_matrix.append(
                {
                    "source_a": a["title"],
                    "source_b": b["title"],
                    "shared_keywords": sorted(overlap)[:15],
                    "similarity_score": round(jaccard, 3),
                }
            )

    return {"coverage_summary": coverage_summary, "comparison_matrix": comparison_matrix}


# --------------------------------------------------------------------------
# Tool 4: fact_check
# --------------------------------------------------------------------------
@mcp.tool()
def fact_check(claim: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Check a claim's support against a list of source documents.

    This is a deterministic lexical-overlap heuristic intended as *evidence*
    for the Fact Verification Agent's LLM-based judgment -- it is not a
    substitute for that judgment. It surfaces which sources most strongly
    support or fail to mention the claim so the LLM can reason over concrete
    evidence rather than relying purely on parametric knowledge.

    Args:
        claim: The factual statement to verify.
        sources: A list of source dicts with "title" and "summary"/"snippet".

    Returns:
        A dict with "confidence_score" (0-1), "supporting_sources", and
        "unsupported" flag.
    """
    claim_keywords = set(re.findall(r"[a-z]{4,}", claim.lower()))
    if not claim_keywords:
        return {"confidence_score": 0.0, "supporting_sources": [], "unsupported": True}

    supporting = []
    for src in sources:
        text = (src.get("summary") or src.get("snippet") or "").lower()
        src_keywords = set(re.findall(r"[a-z]{4,}", text))
        overlap = claim_keywords & src_keywords
        score = len(overlap) / len(claim_keywords)
        if score > 0.25:
            supporting.append(
                {
                    "title": src.get("title", "untitled"),
                    "url": src.get("url", ""),
                    "match_score": round(score, 3),
                    "matched_terms": sorted(overlap)[:10],
                }
            )

    supporting.sort(key=lambda s: s["match_score"], reverse=True)
    confidence = min(1.0, sum(s["match_score"] for s in supporting[:3]) / 3) if supporting else 0.0

    return {
        "claim": claim,
        "confidence_score": round(confidence, 3),
        "supporting_sources": supporting,
        "unsupported": len(supporting) == 0,
    }


# --------------------------------------------------------------------------
# Tool 5: generate_citations
# --------------------------------------------------------------------------
@mcp.tool()
def generate_citations(
    papers: list[dict[str, Any]],
    style: Literal["APA", "MLA", "Chicago"] = "APA",
) -> dict[str, Any]:
    """Generate formatted bibliographic citations for a list of papers.

    Args:
        papers: A list of paper dicts with title, authors, published, url.
        style: Citation style: "APA", "MLA", or "Chicago".

    Returns:
        A dict with "citations", a list of formatted citation strings, and
        the requested "style".
    """
    citations = []
    for p in papers:
        title = p.get("title", "Untitled").rstrip(".")
        authors = p.get("authors") or []
        year = _extract_year(p.get("published", ""))
        url = p.get("url", "")

        if not authors:
            # Sentinel used verbatim -- must NOT be run through name-parsing
            # logic below (that previously misread "Unknown Author" as a
            # literal "Firstname Lastname" pair and mangled it).
            author_str = "Unknown Author"
        elif style == "APA":
            author_str = _format_authors_apa(authors)
        elif style == "MLA":
            author_str = authors[0]
            author_str += " et al." if len(authors) > 1 else ""
        else:  # Chicago
            author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")

        if style == "APA":
            citations.append(f"{author_str} ({year}). {title}. arXiv. {url}")
        elif style == "MLA":
            citations.append(f'{author_str}. "{title}." arXiv, {year}, {url}.')
        else:  # Chicago
            citations.append(f'{author_str}. "{title}." arXiv ({year}). {url}.')

    return {"citations": citations, "style": style, "count": len(citations)}


def _extract_year(published: str) -> str:
    match = re.match(r"(\d{4})", published or "")
    return match.group(1) if match else "n.d."


def _format_authors_apa(authors: list[str]) -> str:
    def last_first(name: str) -> str:
        parts = name.strip().split()
        if len(parts) < 2:
            return name
        return f"{parts[-1]}, {' '.join(p[0] + '.' for p in parts[:-1])}"

    if not authors:
        return "Unknown Author"
    if len(authors) == 1:
        return last_first(authors[0])
    if len(authors) <= 6:
        return ", ".join(last_first(a) for a in authors[:-1]) + f", & {last_first(authors[-1])}"
    return last_first(authors[0]) + " et al."


# --------------------------------------------------------------------------
# Tool 6: export_report
# --------------------------------------------------------------------------
@mcp.tool()
def export_report(
    title: str,
    content: str,
    format: Literal["markdown", "html", "text"] = "markdown",
) -> dict[str, Any]:
    """Export the final research report to disk.

    Args:
        title: The report title, also used to derive a safe filename.
        content: The full report body.
        format: Output format: "markdown", "html", or "text".

    Returns:
        A dict with "file_path" and "format" on success, or "error".
    """
    safe_slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", title.strip().lower()).strip("-")[:80] or "report"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    extension = {"markdown": "md", "html": "html", "text": "txt"}[format]
    filename = f"{safe_slug}-{timestamp}.{extension}"

    out_dir = settings.reports_dir
    file_path = (out_dir / filename).resolve()

    # Defense in depth: ensure the resolved path never escapes the reports dir,
    # even though the filename was already sanitized above.
    if out_dir.resolve() not in file_path.parents:
        return {"error": "Resolved report path escaped the reports directory."}

    try:
        if format == "html":
            body = f"<html><head><title>{html.escape(title)}</title></head><body>" \
                   f"<h1>{html.escape(title)}</h1><pre>{html.escape(content)}</pre></body></html>"
        elif format == "markdown":
            body = f"# {title}\n\n{content}\n"
        else:
            body = f"{title}\n{'=' * len(title)}\n\n{content}\n"

        file_path.write_text(body, encoding="utf-8")
    except OSError as exc:
        return {"error": f"Failed to write report: {exc}"}

    return {"file_path": str(file_path), "format": format, "bytes_written": len(body)}


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------
def main() -> None:
    """Run the MCP server using the transport configured in settings."""
    transport = settings.mcp_transport
    logger.info("Starting research-tools MCP server via transport=%s", transport)
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "streamable-http":
        mcp.run(transport="streamable-http")
    elif transport == "sse":
        mcp.run(transport="sse")
    else:
        raise ValueError(f"Unsupported MCP_TRANSPORT: {transport}")


if __name__ == "__main__":
    main()
