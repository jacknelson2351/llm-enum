from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent


class LLMBackend(str, Enum):
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"


class Settings(BaseSettings):
    backend: LLMBackend = LLMBackend.OLLAMA

    ollama_base_url: str = "http://localhost:11434"
    lmstudio_base_url: str = "http://localhost:1234"
    ollama_model: str = "llama3:latest"
    lmstudio_model: str = ""

    ollama_timeout: float = 30.0
    analysis_temperature: float = 0.0
    suggestion_temperature: float = 0.3
    max_analysis_tokens: int = 256
    max_suggestion_tokens: int = 512

    db_path: str = str(BASE_DIR / "agent_enum.db")
    host: str = "127.0.0.1"
    port: int = 8765

    bisect_max_iterations: int = 8
    bisect_min_words: int = 3

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @field_validator("db_path", mode="before")
    @classmethod
    def _resolve_db_path(cls, value: str) -> str:
        if value is None:
            return str(BASE_DIR / "agent_enum.db")
        path = Path(value)
        if not path.is_absolute():
            path = BASE_DIR / path
        return str(path)

    @property
    def active_base_url(self) -> str:
        if self.backend == LLMBackend.OLLAMA:
            return self.ollama_base_url
        return self.lmstudio_base_url

    @property
    def active_model(self) -> str:
        if self.backend == LLMBackend.OLLAMA:
            return self.ollama_model
        return self.lmstudio_model


settings = Settings()
