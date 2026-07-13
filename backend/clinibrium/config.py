"""Runtime configuration: Settings read from environment variables (pydantic-settings)."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    DATABASE_URL: str = "postgresql://clinibrium:clinibrium@localhost:5432/clinibrium"
    ANTHROPIC_API_KEY: str = ""

    # Track B (ML) — optional. If None, `ml_client.predict()` degrades to
    # None immediately and pipeline A keeps working without B (INV-6).
    ML_PREDICT_URL: str | None = None
    ML_PREDICT_TIMEOUT_S: float = 2.0

    # Audit persistence — best-effort JSONL fallback when Postgres is unavailable.
    AUDIT_LOG_PATH: str = "./audit_events.jsonl"

    # AD-6 / hard rule 2: recording_mode is server-side. It is NEVER taken
    # from the request body — it is set here (env / .env) and passed to the
    # orchestrator. Default False: ambulatory uses Haiku 4.5 (cost); True
    # forces Opus 4.8 for demos / gold-set evaluation.
    RECORDING_MODE: bool = False

    # P1.4: gate for demo-only / debug features (e.g. the "Kill Claude"
    # ?debug_kill_reasoner backdoor). Default False → those features return 403
    # in a normal/public deployment. Enable for the demo or recording.
    DEMO_MODE: bool = False

    # Comma-separated list of browser origins allowed by CORS. The default
    # keeps the local demo working unchanged; a public deploy sets this to
    # the frontend's public URL (e.g. "https://clinibrium.up.railway.app").
    CORS_ORIGINS: str = "http://localhost:3000"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
