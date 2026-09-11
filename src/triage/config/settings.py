"""Configuração de runtime carregada de variáveis de ambiente / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuração da aplicação, prefixada por TRIAGE_."""

    model_config = SettingsConfigDict(
        env_prefix="TRIAGE_", env_file=".env", extra="ignore"
    )

    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    backend: Literal["sklearn", "onnx"] = "sklearn"
    model_name: str = "logistic_regression"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Retorna uma instância cacheada de Settings."""
    return Settings()
