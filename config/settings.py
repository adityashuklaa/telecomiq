"""Centralized configuration and logging for TelecomIQ.

All tunables come from environment variables (loaded from `.env` in local dev).
No secrets are hardcoded — see `.env.example`.
"""
from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, populated from the environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Anthropic / Claude
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-4-8"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "telecomiq"
    postgres_user: str = "telecomiq"
    postgres_password: str = "telecomiq"

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_cdr: str = "cdr-stream"
    kafka_topic_network: str = "network-stream"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    # App
    log_level: str = "INFO"
    model_dir: Path = Path("artifacts")
    data_dir: Path = Path("data/generated")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    settings = Settings()
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings


def configure_logging(name: str) -> logging.Logger:
    """Return a configured logger that writes structured-ish lines to stdout."""
    settings = get_settings()
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(settings.log_level.upper())
    logger.propagate = False
    return logger
