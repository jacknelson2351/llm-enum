from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent


class LLMBackend(str, Enum):
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class Settings(BaseSettings):
    backend: LLMBackend = LLMBackend.OLLAMA

    ollama_base_url: str = "http://localhost:11434"
    lmstudio_base_url: str = "http://localhost:1234"
    openai_base_url: str = "https://api.openai.com"
    anthropic_base_url: str = "https://api.anthropic.com"
    google_base_url: str = "https://generativelanguage.googleapis.com"

    ollama_model: str = "llama3:latest"
    lmstudio_model: str = ""
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-4-20250514"
    google_model: str = "gemini-2.0-flash"

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    ollama_timeout: float = 30.0
    analysis_temperature: float = 0.0
    suggestion_temperature: float = 0.6
    max_analysis_tokens: int = 256
    max_suggestion_tokens: int = 2048

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
        return {
            LLMBackend.OLLAMA: self.ollama_base_url,
            LLMBackend.LMSTUDIO: self.lmstudio_base_url,
            LLMBackend.OPENAI: self.openai_base_url,
            LLMBackend.ANTHROPIC: self.anthropic_base_url,
            LLMBackend.GOOGLE: self.google_base_url,
        }[self.backend]

    @property
    def active_model(self) -> str:
        return {
            LLMBackend.OLLAMA: self.ollama_model,
            LLMBackend.LMSTUDIO: self.lmstudio_model,
            LLMBackend.OPENAI: self.openai_model,
            LLMBackend.ANTHROPIC: self.anthropic_model,
            LLMBackend.GOOGLE: self.google_model,
        }[self.backend]

    @property
    def active_api_key(self) -> str:
        return {
            LLMBackend.OLLAMA: "",
            LLMBackend.LMSTUDIO: "",
            LLMBackend.OPENAI: self.openai_api_key,
            LLMBackend.ANTHROPIC: self.anthropic_api_key,
            LLMBackend.GOOGLE: self.google_api_key,
        }[self.backend]


settings = Settings()
