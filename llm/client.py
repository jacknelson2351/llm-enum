from __future__ import annotations

import re
import time

import httpx

from config import LLMBackend
from llm.runtime_config import runtime_cfg

_THINK_TAG_RE = re.compile(
    r"<(?:think|thinking)>\s*(.*?)\s*</(?:think|thinking)>",
    re.IGNORECASE | re.DOTALL,
)


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


def _coerce_text_content(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    parts.append(cleaned)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
                continue
            nested = _coerce_text_content(item.get("content"))
            if nested:
                parts.append(nested)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        for key in ("text", "output_text", "thinking", "reasoning", "reasoning_content"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
        for key in ("content", "output", "parts"):
            nested = _coerce_text_content(value.get(key))
            if nested:
                return nested
    return ""


def _strip_thinking_blocks(text: str) -> tuple[str, str]:
    if not text:
        return "", ""
    reasoning_parts = [match.group(1).strip() for match in _THINK_TAG_RE.finditer(text) if match.group(1).strip()]
    cleaned = _THINK_TAG_RE.sub("", text).strip()
    return cleaned, "\n\n".join(reasoning_parts).strip()


def _split_openai_compat_content(value: object) -> tuple[str, str]:
    if isinstance(value, str):
        return _strip_thinking_blocks(value)
    if isinstance(value, list):
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for item in value:
            item_text, item_reasoning = _split_openai_compat_content(item)
            if item_text:
                text_parts.append(item_text)
            if item_reasoning:
                reasoning_parts.append(item_reasoning)
        return "\n".join(text_parts).strip(), "\n\n".join(reasoning_parts).strip()
    if isinstance(value, dict):
        kind = str(value.get("type", "") or "").lower()
        explicit_reasoning = _coerce_text_content(
            value.get("thinking")
            or value.get("reasoning")
            or value.get("reasoning_content")
        )
        if kind in {"reasoning", "thinking"}:
            return "", explicit_reasoning or _coerce_text_content(value.get("text") or value.get("content"))
        text_value = value.get("text")
        if text_value is None:
            text_value = value.get("output_text")
        if text_value is None:
            text_value = value.get("content")
        if text_value is None:
            text_value = value.get("parts")
        text = _coerce_text_content(text_value)
        text, inline_reasoning = _strip_thinking_blocks(text)
        reasoning_parts = [part for part in (explicit_reasoning, inline_reasoning) if part]
        return text, "\n\n".join(reasoning_parts).strip()
    return "", ""


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
        self.last_reasoning = ""

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

    def _extract_openai_compat_text(self, data: dict) -> str:
        choices = list(data.get("choices", []) or [])
        if not choices:
            raise ValueError("Empty response from model")

        choice = choices[0]
        message = choice.get("message", {}) or {}
        content, content_reasoning = _split_openai_compat_content(message.get("content"))
        reasoning = _coerce_text_content(
            message.get("reasoning_content")
            or message.get("reasoning")
            or choice.get("reasoning")
        )
        if not reasoning:
            reasoning = content_reasoning
        self.last_reasoning = reasoning

        if not content:
            content = _coerce_text_content(choice.get("text") or choice.get("output_text"))
            content, inline_reasoning = _strip_thinking_blocks(content)
            if not self.last_reasoning:
                self.last_reasoning = inline_reasoning

        if not content:
            refusal = _coerce_text_content(message.get("refusal") or choice.get("refusal"))
            if refusal:
                return refusal
            raise ValueError("Empty response from model")
        return content

    def _retry_openai_compat_body(
        self,
        *,
        body: dict,
        system: str,
        user: str,
        error_text: str,
        max_tokens: int,
    ) -> bool:
        detail = error_text.lower()
        changed = False

        if "temperature" in detail:
            changed = body.pop("temperature", None) is not None or changed
        if "max_completion_tokens" in detail:
            body.pop("max_completion_tokens", None)
            body["max_tokens"] = max_tokens
            changed = True
        if "max_tokens" in detail and "unsupported" in detail:
            changed = body.pop("max_tokens", None) is not None or changed
        if "system" in detail and "role" in detail:
            body["messages"] = [
                {"role": "user", "content": f"SYSTEM INSTRUCTIONS:\n{system}\n\nUSER REQUEST:\n{user}"}
            ]
            changed = True
        if "developer" in detail and "role" in detail:
            body["messages"] = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            changed = True
        return changed

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
        # Auto-fix unsupported parameters for OpenAI-compatible servers.
        if self.backend in {LLMBackend.OPENAI, LLMBackend.LMSTUDIO} and resp.status_code == 400:
            error_text = _error_detail(resp)
            if self._retry_openai_compat_body(
                body=body,
                system=system,
                user=user,
                error_text=error_text,
                max_tokens=max_tokens,
            ):
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                    headers=headers,
                )
                _raise_with_detail(resp)
            else:
                raise httpx.HTTPStatusError(
                    f"{resp.status_code}: {error_text}",
                    request=resp.request,
                    response=resp,
                )

        _raise_with_detail(resp)
        data = resp.json()
        return self._extract_openai_compat_text(data)

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
