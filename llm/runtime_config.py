from __future__ import annotations

from dataclasses import dataclass, field

from config import LLMBackend, settings


@dataclass
class RuntimeConfig:
    backend: LLMBackend = LLMBackend.OLLAMA
    ollama_url: str = "http://localhost:11434"
    lmstudio_url: str = "http://localhost:1234"
    openai_url: str = "https://api.openai.com"
    anthropic_url: str = "https://api.anthropic.com"
    google_url: str = "https://generativelanguage.googleapis.com"
    model: str = ""
    api_key: str = ""

    @property
    def active_url(self) -> str:
        return {
            LLMBackend.OLLAMA: self.ollama_url,
            LLMBackend.LMSTUDIO: self.lmstudio_url,
            LLMBackend.OPENAI: self.openai_url,
            LLMBackend.ANTHROPIC: self.anthropic_url,
            LLMBackend.GOOGLE: self.google_url,
        }[self.backend]


runtime_cfg = RuntimeConfig(
    backend=settings.backend,
    ollama_url=settings.ollama_base_url,
    lmstudio_url=settings.lmstudio_base_url,
    openai_url=settings.openai_base_url,
    anthropic_url=settings.anthropic_base_url,
    google_url=settings.google_base_url,
    model=settings.active_model,
    api_key=settings.active_api_key,
)
