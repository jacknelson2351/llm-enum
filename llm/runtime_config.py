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
    def model_name(self) -> str:
        return (self.model or "").strip()

    @property
    def model_name_lower(self) -> str:
        return self.model_name.lower()

    @property
    def is_local_backend(self) -> bool:
        return self.backend in {LLMBackend.OLLAMA, LLMBackend.LMSTUDIO}

    @property
    def is_openai_compatible_backend(self) -> bool:
        return self.backend in {LLMBackend.OPENAI, LLMBackend.LMSTUDIO}

    @property
    def prefers_compact_context(self) -> bool:
        name = self.model_name_lower
        return self.is_local_backend or any(
            marker in name
            for marker in ("gpt-oss", "deepseek-r1", "qwq", "qwen", "llama", "mistral")
        )

    @property
    def prefers_tight_loops(self) -> bool:
        return self.prefers_compact_context

    @property
    def supports_reasoning_parser(self) -> bool:
        return self.is_openai_compatible_backend or self.backend == LLMBackend.OLLAMA

    @property
    def active_url(self) -> str:
        return {
            LLMBackend.OLLAMA: self.ollama_url,
            LLMBackend.LMSTUDIO: self.lmstudio_url,
            LLMBackend.OPENAI: self.openai_url,
            LLMBackend.ANTHROPIC: self.anthropic_url,
            LLMBackend.GOOGLE: self.google_url,
        }[self.backend]

    def analysis_token_budget(self, fallback: int) -> int:
        return min(fallback, 160) if self.prefers_compact_context else fallback

    def pipeline_token_budget(self, fallback: int) -> int:
        return min(fallback, 220) if self.prefers_compact_context else fallback

    def reconstruction_token_budget(self, fallback: int) -> int:
        return min(fallback, 420) if self.prefers_compact_context else min(fallback, 960)

    def strategy_planner_token_budget(self, fallback: int) -> int:
        return min(fallback, 540) if self.prefers_compact_context else min(fallback, 960)

    def strategy_writer_token_budget(self, fallback: int) -> int:
        return min(fallback, 900) if self.prefers_compact_context else min(fallback, 1400)

    def compiled_brief_char_budget(self) -> int:
        return 2200 if self.prefers_compact_context else 5200


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
