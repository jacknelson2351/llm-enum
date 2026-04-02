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


class OpenAICompatParsingTests(unittest.TestCase):
    def test_extract_openai_compat_text_strips_think_blocks_and_keeps_reasoning(self) -> None:
        client = LLMClient(
            backend=LLMBackend.LMSTUDIO,
            base_url="http://localhost:1234",
            model="gpt-oss-20b",
        )

        output = client._extract_openai_compat_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": "<think>map the topology first</think>\nFinal answer",
                            "reasoning_content": "planner summary",
                        }
                    }
                ]
            }
        )

        self.assertEqual(output, "Final answer")
        self.assertEqual(client.last_reasoning, "planner summary")

    def test_extract_openai_compat_text_reads_structured_content_arrays(self) -> None:
        client = LLMClient(
            backend=LLMBackend.LMSTUDIO,
            base_url="http://localhost:1234",
            model="gpt-oss-20b",
        )

        output = client._extract_openai_compat_text(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "reasoning", "thinking": "plan"},
                                {"type": "output_text", "text": "Compact result"},
                            ]
                        }
                    }
                ]
            }
        )

        self.assertEqual(output, "Compact result")
        self.assertEqual(client.last_reasoning, "plan")
