from __future__ import annotations

import unittest

from config import LLMBackend
from llm.client import LLMClient


class _FakeResponse:
    is_success = True
    status_code = 200
    text = ""
    request = None

    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeGoogleClient:
    def __init__(self, payload: dict):
        self.payload = payload

    async def post(self, *_args, **_kwargs):
        return _FakeResponse(self.payload)


class GoogleClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_google_joins_all_text_parts(self) -> None:
        client = LLMClient(
            backend=LLMBackend.GOOGLE,
            base_url="https://example.invalid",
            model="gemini-2.0-flash",
            api_key="test-key",
        )
        http_client = _FakeGoogleClient({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "```json\n["},
                            {"text": "{\"objective\":\"one\"},"},
                            {"text": "{\"objective\":\"two\"}\n]```"},
                        ]
                    }
                }
            ]
        })

        output = await client._chat_google(
            http_client,
            system="Return JSON only.",
            user="Generate strategies.",
            temperature=0.0,
            max_tokens=256,
        )

        self.assertEqual(
            output,
            "```json\n[{\"objective\":\"one\"},{\"objective\":\"two\"}\n]```",
        )
