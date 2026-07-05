# ==============================================================================
# AI Research Assistant - Makefile
# ==============================================================================

PYTHON ?= python3
VENV ?= .venv
PIP ?= $(VENV)/bin/pip
PY ?= $(VENV)/bin/python
UVICORN ?= $(VENV)/bin/uvicorn

.PHONY: help venv install install-dev env run-mcp run-api run-adk-web run-adk-cli test lint format docker-build docker-run docker-stop clean

help:
	@echo "AI Research Assistant - available targets:"
	@echo "  make venv          Create a virtual environment in $(VENV)"
	@echo "  make install       Install runtime dependencies"
	@echo "  make install-dev   Install runtime + dev dependencies"
	@echo "  make env           Copy .env.example to .env (won't overwrite an existing .env)"
	@echo "  make run-mcp       Run the MCP tool server standalone (stdio transport)"
	@echo "  make run-api       Run the FastAPI app with uvicorn (auto-reload)"
	@echo "  make run-adk-web   Run ADK's built-in web UI against this project"
	@echo "  make run-adk-cli   Run ADK's built-in CLI chat against this project"
	@echo "  make test          Run the test suite"
	@echo "  make lint          Run ruff + mypy"
	@echo "  make format        Auto-format code with black + ruff"
	@echo "  make docker-build  Build the production Docker image"
	@echo "  make docker-run    Run the Docker image, mapping port 8080"
	@echo "  make clean         Remove caches, build artifacts, and the venv"

venv:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -e .

install-dev: venv
	$(PIP) install -e ".[dev]"

env:
	@test -f .env || cp .env.example .env
	@echo "Edit .env with your GOOGLE_API_KEY and other secrets."

run-mcp:
	$(PY) -m ai_research_assistant.mcp_server

run-api:
	$(UVICORN) ai_research_assistant.fast_api_app:app --host 0.0.0.0 --port 8080 --reload --app-dir src

run-adk-web:
	$(VENV)/bin/adk web src

run-adk-cli:
	$(VENV)/bin/adk run src/ai_research_assistant

test:
	$(VENV)/bin/pytest -v

lint:
	$(VENV)/bin/ruff check src
	$(VENV)/bin/mypy src

format:
	$(VENV)/bin/black src
	$(VENV)/bin/ruff check --fix src

docker-build:
	docker build -t ai-research-assistant:latest .

docker-run:
	docker run --rm -it \
		--env-file .env \
		-p 8080:8080 \
		--name ai-research-assistant \
		ai-research-assistant:latest

docker-stop:
	docker stop ai-research-assistant || true

clean:
	rm -rf $(VENV) build dist *.egg-info src/*.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
