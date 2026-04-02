from __future__ import annotations

import logging

from config import settings
from graph.state import EnumerationState
from llm.client import get_client
from llm.json_repair import extract_json
from prompts.templates import RESPONSE_ANALYZER_SYSTEM, RESPONSE_ANALYZER_USER

log = logging.getLogger(__name__)


async def response_analyzer(state: EnumerationState) -> EnumerationState:
    client = get_client(timeout=settings.ollama_timeout)
    prompt = RESPONSE_ANALYZER_USER.format(
        probe_text=state["probe_text"],
        response_text=state["response_text"],
    )
    try:
        raw = await client.chat(
            system=RESPONSE_ANALYZER_SYSTEM,
            user=prompt,
            temperature=settings.analysis_temperature,
            max_tokens=settings.max_analysis_tokens,
        )
        data = extract_json(raw)
        classification = data.get("classification", "NEUTRAL").upper()
        if classification not in ("LEAK", "REFUSAL", "TOOL_DISCLOSURE", "NEUTRAL"):
            classification = "NEUTRAL"
        return {
            **state,
            "classification": classification,
            "analysis_confidence": float(data.get("confidence", 0.5)),
            "analysis_reasoning": data.get("reasoning", ""),
            "events": state.get("events", []) + [{
                "type": "analysis_complete",
                "data": {
                    "classification": classification,
                    "confidence": float(data.get("confidence", 0.5)),
                    "reasoning": data.get("reasoning", ""),
                },
            }],
        }
    except Exception as e:
        log.error("ResponseAnalyzer failed: %s", e)
        return {
            **state,
            "classification": "NEUTRAL",
            "analysis_confidence": 0.0,
            "analysis_reasoning": f"Analysis error: {e}",
            "error": str(e),
            "events": state.get("events", []) + [{
                "type": "error",
                "data": {"message": f"Analysis failed: {e}"},
            }],
        }
