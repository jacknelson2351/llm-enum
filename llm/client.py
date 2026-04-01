from __future__ import annotations

import time

import httpx

from config import LLMBackend
from llm.runtime_config import runtime_cfg


class LLMClient:
    def __init__(
        self,
        backend: LLMBackend | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ):
        self.backend = backend or runtime_cfg.backend
        self.base_url = (base_url or runtime_cfg.active_url).rstrip("/")
        self.model = model or runtime_cfg.model
        self.timeout = timeout

    async def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if self.backend == LLMBackend.OLLAMA:
                resp = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        },
                    },
                )
                resp.raise_for_status()
                return resp.json()["message"]["content"]
            else:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "stream": False,
                    },
                    headers={"Authorization": "Bearer lm-studio"},
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if self.backend == LLMBackend.OLLAMA:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                return [m["name"] for m in resp.json().get("models", [])]
            else:
                resp = await client.get(
                    f"{self.base_url}/v1/models",
                    headers={"Authorization": "Bearer lm-studio"},
                )
                resp.raise_for_status()
                return [m["id"] for m in resp.json().get("data", [])]

    async def health_check(self) -> tuple[bool, str, int]:
        start = time.monotonic()
        try:
            models = await self.list_models()
            latency = int((time.monotonic() - start) * 1000)
            if not models:
                return False, "Connected but no models loaded", latency
            return True, f"OK — {len(models)} model(s) available", latency
        except httpx.ConnectError:
            latency = int((time.monotonic() - start) * 1000)
            return False, f"Connection refused at {self.base_url}", latency
        except Exception as e:
            latency = int((time.monotonic() - start) * 1000)
            return False, str(e), latency


def get_client(timeout: float = 30.0) -> LLMClient:
    return LLMClient(timeout=timeout)
