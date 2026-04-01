from __future__ import annotations

import json
import logging

from config import settings
from graph.state import EnumerationState
from llm.client import get_client
from prompts.templates import (
    FRAGMENT_EXTRACTOR_SYSTEM,
    FRAGMENT_EXTRACTOR_USER,
    KNOWLEDGE_UPDATER_SYSTEM,
    KNOWLEDGE_UPDATER_USER,
)

log = logging.getLogger(__name__)


async def fragment_extractor(state: EnumerationState) -> EnumerationState:
    client = get_client(timeout=settings.ollama_timeout)
    prompt = FRAGMENT_EXTRACTOR_USER.format(
        probe_text=state["probe_text"],
        response_text=state["response_text"],
    )
    try:
        raw = await client.chat(
            system=FRAGMENT_EXTRACTOR_SYSTEM,
            user=prompt,
            temperature=settings.analysis_temperature,
            max_tokens=settings.max_analysis_tokens,
        )
        data = json.loads(raw)
        return {
            **state,
            "fragment_text": data.get("fragment", ""),
            "fragment_confidence": float(data.get("confidence", 0.5)),
            "fragment_position": data.get("position_hint", "unknown"),
            "events": state.get("events", []) + [{
                "type": "fragment_found",
                "data": {
                    "fragment": data.get("fragment", ""),
                    "confidence": float(data.get("confidence", 0.5)),
                    "position_hint": data.get("position_hint", "unknown"),
                },
            }],
        }
    except Exception as e:
        log.error("FragmentExtractor failed: %s", e)
        return {
            **state,
            "fragment_text": "",
            "fragment_confidence": 0.0,
            "fragment_position": "unknown",
            "error": str(e),
        }


async def knowledge_updater(state: EnumerationState) -> EnumerationState:
    client = get_client(timeout=settings.ollama_timeout)
    prompt = KNOWLEDGE_UPDATER_USER.format(
        probe_text=state["probe_text"],
        response_text=state["response_text"],
    )
    try:
        raw = await client.chat(
            system=KNOWLEDGE_UPDATER_SYSTEM,
            user=prompt,
            temperature=settings.analysis_temperature,
            max_tokens=settings.max_analysis_tokens,
        )
        data = json.loads(raw)
        return {
            **state,
            "new_knowledge": {
                "tools": data.get("tools", []),
                "constraints": data.get("constraints", []),
                "persona": data.get("persona", []),
                "raw_facts": data.get("raw_facts", []),
            },
            "events": state.get("events", []) + [{
                "type": "analysis_complete",
                "data": {"classification": "TOOL_DISCLOSURE", "knowledge": data},
            }],
        }
    except Exception as e:
        log.error("KnowledgeUpdater failed: %s", e)
        return {**state, "new_knowledge": {}, "error": str(e)}
