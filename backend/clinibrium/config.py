"""Configuración runtime: Settings leídos de variables de entorno (pydantic-settings)."""
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

    # Track B (ML) — opcional. Si es None, `ml_client.predict()` degrada a
    # None inmediatamente y el pipeline A sigue funcionando sin B (INV-6).
    ML_PREDICT_URL: str | None = None
    ML_PREDICT_TIMEOUT_S: float = 2.0

    # Audit persistence — best-effort JSONL fallback cuando Postgres no está.
    AUDIT_LOG_PATH: str = "./audit_events.jsonl"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
