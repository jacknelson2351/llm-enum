from __future__ import annotations

from dataclasses import dataclass

from config import LLMBackend, settings


@dataclass
class RuntimeConfig:
    backend: LLMBackend = LLMBackend.OLLAMA
    ollama_url: str = "http://localhost:11434"
    lmstudio_url: str = "http://localhost:1234"
    model: str = ""

    @property
    def active_url(self) -> str:
        if self.backend == LLMBackend.OLLAMA:
            return self.ollama_url
        return self.lmstudio_url


runtime_cfg = RuntimeConfig(
    backend=settings.backend,
    ollama_url=settings.ollama_base_url,
    lmstudio_url=settings.lmstudio_base_url,
    model=settings.active_model,
)
