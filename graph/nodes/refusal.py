from __future__ import annotations

import logging
from textwrap import shorten

from config import settings
from graph.progress import emit_agent_step
from graph.state import EnumerationState
from llm.client import get_client
from llm.json_repair import extract_json
from llm.runtime_config import runtime_cfg
from prompts.templates import REFUSAL_CLASSIFIER_SYSTEM, REFUSAL_CLASSIFIER_USER

log = logging.getLogger(__name__)


async def refusal_classifier(state: EnumerationState) -> EnumerationState:
    await emit_agent_step(
        state,
        phase="refusal",
        label="Mapping refusal boundary",
        status="start",
        detail="Identifying the refusal family and likely trigger.",
    )
    client = get_client(timeout=settings.ollama_timeout)
    prompt = REFUSAL_CLASSIFIER_USER.format(
        probe_text=state["probe_text"],
        response_text=state["response_text"],
    )
    try:
        raw = await client.chat(
            system=REFUSAL_CLASSIFIER_SYSTEM,
            user=prompt,
            temperature=settings.analysis_temperature,
            max_tokens=runtime_cfg.analysis_token_budget(settings.max_analysis_tokens),
        )
        data = extract_json(raw)
        refusal_type = data.get("refusal_type", "hard_refusal")
        trigger = data.get("trigger_candidate", "")
        await emit_agent_step(
            state,
            phase="refusal",
            label="Mapping refusal boundary",
            status="done",
            detail=shorten(trigger or client.last_reasoning or refusal_type, width=120, placeholder="..."),
        )
        return {
            **state,
            "refusal_type": refusal_type,
            "trigger_candidate": trigger,
            "refusal_confidence": float(data.get("confidence", 0.5)),
            "events": state.get("events", []) + [{
                "type": "refusal_classified",
                "data": {
                    "refusal_type": refusal_type,
                    "trigger_candidate": trigger,
                    "confidence": float(data.get("confidence", 0.5)),
                    "reasoning": data.get("reasoning", ""),
                },
            }],
        }
    except Exception as e:
        log.error("RefusalClassifier failed: %s", e)
        await emit_agent_step(
            state,
            phase="refusal",
            label="Mapping refusal boundary",
            status="error",
            detail=str(e),
        )
        return {
            **state,
            "refusal_type": "hard_refusal",
            "trigger_candidate": "",
            "refusal_confidence": 0.0,
            "error": str(e),
        }
