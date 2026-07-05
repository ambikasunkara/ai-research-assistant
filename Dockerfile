# ==============================================================================
# AI Research Assistant - Production Dockerfile
# Multi-stage build: install deps in a builder layer, copy a slim runtime.
# ==============================================================================

# ---- Builder stage -----------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed only to build wheels (kept out of the final image).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install .

# ---- Runtime stage -------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Run as a non-root user for defense in depth.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /build/src /app/src

# Reports and logs directories are created at runtime, but pre-create with
# correct ownership so the non-root user can write to them.
RUN mkdir -p /app/reports /app/logs && chown -R appuser:appuser /app

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    REPORTS_OUTPUT_DIR=/app/reports \
    AUDIT_LOG_DIR=/app/logs \
    API_HOST=0.0.0.0 \
    API_PORT=8080

USER appuser
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health', timeout=3).status == 200 else 1)"

# The FastAPI layer is the container's public interface. It internally
# spawns the MCP server as a stdio subprocess per McpToolset (see agent.py),
# so no separate MCP container/process is required for this deployment mode.
CMD ["python", "-m", "uvicorn", "ai_research_assistant.fast_api_app:app", "--host", "0.0.0.0", "--port", "8080"]
