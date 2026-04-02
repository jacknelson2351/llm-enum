from __future__ import annotations

import time

import httpx

from config import LLMBackend
from llm.runtime_config import runtime_cfg


def _error_detail(resp: httpx.Response) -> str:
    """Extract a human-readable error message from an API error response."""
    text = resp.text or ""
    try:
        body = resp.json()
        msg = body.get("error", {})
        if isinstance(msg, dict):
            msg = msg.get("message", "")
        if msg:
            return str(msg)
    except Exception:
        pass
    return text[:300] or f"HTTP {resp.status_code} (empty body)"


def _raise_with_detail(resp: httpx.Response) -> None:
    """Like raise_for_status but includes the API error body."""
    if resp.is_success:
        return
    detail = _error_detail(resp)
    raise httpx.HTTPStatusError(
        f"{resp.status_code}: {detail}",
        request=resp.request,
        response=resp,
    )


class LLMClient:
    def __init__(
        self,
        backend: LLMBackend | None = None,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self.backend = backend or runtime_cfg.backend
        self.base_url = (base_url or runtime_cfg.active_url).rstrip("/")
        self.model = model or runtime_cfg.model
        self.api_key = api_key or runtime_cfg.api_key
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
                return await self._chat_ollama(client, system, user, temperature, max_tokens)
            elif self.backend == LLMBackend.ANTHROPIC:
                return await self._chat_anthropic(client, system, user, temperature, max_tokens)
            elif self.backend == LLMBackend.GOOGLE:
                return await self._chat_google(client, system, user, temperature, max_tokens)
            else:
                # LMSTUDIO, OPENAI, and any OpenAI-compatible endpoint
                return await self._chat_openai_compat(client, system, user, temperature, max_tokens)

    async def _chat_ollama(
        self, client: httpx.AsyncClient, system: str, user: str, temperature: float, max_tokens: int
    ) -> str:
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
        _raise_with_detail(resp)
        return resp.json()["message"]["content"]

    def _is_openai_reasoning(self) -> bool:
        """Check if the current model is an OpenAI reasoning model (o-series)."""
        model = (self.model or "").lower()
        return self.backend == LLMBackend.OPENAI and any(
            model.startswith(p) for p in ("o1", "o3", "o4")
        )

    async def _chat_openai_compat(
        self, client: httpx.AsyncClient, system: str, user: str, temperature: float, max_tokens: int
    ) -> str:
        headers = {}
        if self.backend == LLMBackend.OPENAI:
            headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.backend == LLMBackend.LMSTUDIO:
            headers["Authorization"] = "Bearer lm-studio"

        reasoning = self._is_openai_reasoning()

        # Reasoning models (o1/o3/o4) use "developer" role instead of "system"
        system_role = "developer" if reasoning else "system"
        messages = [
            {"role": system_role, "content": system},
            {"role": "user", "content": user},
        ]

        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        if self.backend == LLMBackend.OPENAI:
            body["max_completion_tokens"] = max_tokens
            # Reasoning models don't support temperature at all
            if not reasoning:
                body["temperature"] = temperature
        else:
            body["max_tokens"] = max_tokens
            body["temperature"] = temperature

        resp = await client.post(
            f"{self.base_url}/v1/chat/completions",
            json=body,
            headers=headers,
        )
        # Auto-fix unsupported parameters: some OpenAI models reject
        # temperature, max_completion_tokens, or system role.
        if self.backend == LLMBackend.OPENAI and resp.status_code == 400:
            # Read the error once and cache it — httpx responses can
            # only be read once reliably.
            error_text = _error_detail(resp)
            detail = error_text.lower()
            changed = False
            if "temperature" in detail:
                body.pop("temperature", None)
                changed = True
            if "max_completion_tokens" in detail:
                body.pop("max_completion_tokens", None)
                body["max_tokens"] = max_tokens
                changed = True
            if "system" in detail and "role" in detail:
                body["messages"] = [
                    {"role": "developer", "content": system},
                    {"role": "user", "content": user},
                ]
                changed = True
            if changed:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                    headers=headers,
                )
                _raise_with_detail(resp)
            else:
                # Not a fixable parameter issue — raise with the cached error
                raise httpx.HTTPStatusError(
                    f"{resp.status_code}: {error_text}",
                    request=resp.request,
                    response=resp,
                )

        _raise_with_detail(resp)
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"].get("content")
        # Some models return null content with a refusal
        if not content:
            refusal = choice["message"].get("refusal") or ""
            if refusal:
                return refusal
            raise ValueError("Empty response from model")
        return content

    async def _chat_anthropic(
        self, client: httpx.AsyncClient, system: str, user: str, temperature: float, max_tokens: int
    ) -> str:
        resp = await client.post(
            f"{self.base_url}/v1/messages",
            json={
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "temperature": temperature,
            },
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        _raise_with_detail(resp)
        content = resp.json().get("content", [])
        return content[0]["text"] if content else ""

    async def _chat_google(
        self, client: httpx.AsyncClient, system: str, user: str, temperature: float, max_tokens: int
    ) -> str:
        resp = await client.post(
            f"{self.base_url}/v1beta/models/{self.model}:generateContent",
            params={"key": self.api_key},
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            },
            headers={"content-type": "application/json"},
        )
        _raise_with_detail(resp)
        candidates = resp.json().get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            chunks = [
                part.get("text", "")
                for part in parts
                if isinstance(part, dict) and part.get("text")
            ]
            return "".join(chunks)
        return ""

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            if self.backend == LLMBackend.OLLAMA:
                resp = await client.get(f"{self.base_url}/api/tags")
                _raise_with_detail(resp)
                return [m["name"] for m in resp.json().get("models", [])]
            elif self.backend == LLMBackend.ANTHROPIC:
                # Anthropic doesn't have a list-models endpoint; return common models
                return [
                    "claude-opus-4-20250514",
                    "claude-sonnet-4-20250514",
                    "claude-haiku-4-20250414",
                ]
            elif self.backend == LLMBackend.GOOGLE:
                resp = await client.get(
                    f"{self.base_url}/v1beta/models",
                    params={"key": self.api_key},
                )
                _raise_with_detail(resp)
                return [
                    m["name"].removeprefix("models/")
                    for m in resp.json().get("models", [])
                    if "generateContent" in ",".join(m.get("supportedGenerationMethods", []))
                ]
            elif self.backend == LLMBackend.OPENAI:
                resp = await client.get(
                    f"{self.base_url}/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                _raise_with_detail(resp)
                return sorted(m["id"] for m in resp.json().get("data", []))
            else:
                # LM Studio
                resp = await client.get(
                    f"{self.base_url}/v1/models",
                    headers={"Authorization": "Bearer lm-studio"},
                )
                _raise_with_detail(resp)
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
