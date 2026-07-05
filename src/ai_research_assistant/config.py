"""
config.py
=========
Centralized, typed configuration for the AI Research Assistant.

Everything that varies between environments (dev / staging / prod) or that is
security-sensitive (API keys) lives here and is sourced from environment
variables / a local `.env` file via `pydantic-settings`. No other module
should read `os.environ` directly -- they should import `settings` from here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (src/ai_research_assistant/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Strongly typed application settings, loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Google Gen AI credentials -----------------------------------
    google_genai_use_vertexai: bool = Field(default=False, alias="GOOGLE_GENAI_USE_VERTEXAI")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    google_cloud_project: str = Field(default="", alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str = Field(default="us-central1", alias="GOOGLE_CLOUD_LOCATION")

    # ---- Models ---------------------------------------------------------
    model_pro: str = Field(default="gemini-2.5-pro", alias="MODEL_PRO")
    model_flash: str = Field(default="gemini-2.5-flash", alias="MODEL_FLASH")

    # ---- MCP server -------------------------------------------------------
    mcp_transport: str = Field(default="stdio", alias="MCP_TRANSPORT")  # stdio | streamable-http
    mcp_server_host: str = Field(default="127.0.0.1", alias="MCP_SERVER_HOST")
    mcp_server_port: int = Field(default=8765, alias="MCP_SERVER_PORT")
    mcp_server_command: str = Field(default="python", alias="MCP_SERVER_COMMAND")
    mcp_server_module: str = Field(
        default="ai_research_assistant.mcp_server", alias="MCP_SERVER_MODULE"
    )

    # ---- Application / API ------------------------------------------------
    app_name: str = Field(default="ai-research-assistant", alias="APP_NAME")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8080, alias="API_PORT")
    allowed_origins: str = Field(default="http://localhost:3000", alias="ALLOWED_ORIGINS")
    research_api_keys: str = Field(default="change-me-dev-key", alias="RESEARCH_API_KEYS")

    # ---- Security checkpoint -----------------------------------------------
    max_input_chars: int = Field(default=4000, alias="MAX_INPUT_CHARS")
    rate_limit_per_minute: int = Field(default=20, alias="RATE_LIMIT_PER_MINUTE")
    block_prompt_injection: bool = Field(default=True, alias="BLOCK_PROMPT_INJECTION")

    # ---- Human approval -----------------------------------------------------
    approval_timeout_seconds: int = Field(default=1800, alias="APPROVAL_TIMEOUT_SECONDS")
    require_human_approval: bool = Field(default=True, alias="REQUIRE_HUMAN_APPROVAL")

    # ---- Audit logging ---------------------------------------------------------
    audit_log_dir: str = Field(default="./logs", alias="AUDIT_LOG_DIR")
    audit_log_file: str = Field(default="audit.log.jsonl", alias="AUDIT_LOG_FILE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ---- Reports -------------------------------------------------------------
    reports_output_dir: str = Field(default="./reports", alias="REPORTS_OUTPUT_DIR")
    max_papers_per_query: int = Field(default=8, alias="MAX_PAPERS_PER_QUERY")

    # ---------------------------------------------------------------------
    @field_validator("mcp_transport")
    @classmethod
    def _validate_transport(cls, v: str) -> str:
        allowed = {"stdio", "streamable-http", "sse"}
        if v not in allowed:
            raise ValueError(f"mcp_transport must be one of {allowed}, got {v!r}")
        return v

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def research_api_keys_set(self) -> set[str]:
        return {k.strip() for k in self.research_api_keys.split(",") if k.strip()}

    @property
    def audit_log_path(self) -> Path:
        d = Path(self.audit_log_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d / self.audit_log_file

    @property
    def reports_dir(self) -> Path:
        d = Path(self.reports_output_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton `Settings` instance."""
    return Settings()


# Convenience module-level instance used throughout the codebase:
#   from ai_research_assistant.config import settings
settings = get_settings()
