from __future__ import annotations

import json
import logging

from config import settings
from graph.nodes.context import (
    classification_count,
    combined_fragments,
    combined_refusals,
    recent_probe_pairs,
    total_probe_count,
)
from graph.state import EnumerationState
from llm.client import get_client
from prompts.templates import PIPELINE_DETECTOR_SYSTEM, PIPELINE_DETECTOR_USER

log = logging.getLogger(__name__)


async def pipeline_detector(state: EnumerationState) -> EnumerationState:
    client = get_client(timeout=settings.ollama_timeout)

    leak_count = classification_count(state, "LEAK")
    refusal_count = classification_count(state, "REFUSAL")
    tool_count = classification_count(state, "TOOL_DISCLOSURE")

    fragment_texts = []
    for fragment in combined_fragments(state):
        text = fragment.get("text", "").strip()
        if text:
            fragment_texts.append(
                f"- [{fragment.get('position_hint', '?')}] {text[:100]}"
            )

    refusal_patterns = []
    for refusal in combined_refusals(state):
        trigger = refusal.get("confirmed_trigger") or refusal.get("trigger_candidate", "?")
        refusal_patterns.append(
            f"- {refusal.get('refusal_type', '?')}: trigger='{trigger}'"
        )

    prev_topo = (
        json.dumps(state.get("pipeline_json", {}), indent=2)
        if state.get("pipeline_json")
        else "None yet"
    )

    prompt = PIPELINE_DETECTOR_USER.format(
        probe_count=total_probe_count(state),
        leak_count=leak_count,
        refusal_count=refusal_count,
        tool_count=tool_count,
        recent_probes=recent_probe_pairs(state),
        fragments="\n".join(fragment_texts) or "None yet",
        refusal_patterns="\n".join(refusal_patterns) or "None yet",
        previous_topology=prev_topo,
    )

    try:
        raw = await client.chat(
            system=PIPELINE_DETECTOR_SYSTEM,
            user=prompt,
            temperature=settings.analysis_temperature,
            max_tokens=settings.max_analysis_tokens,
        )
        data = json.loads(raw)
        return {
            **state,
            "pipeline_json": data,
            "events": state.get("events", []) + [{
                "type": "pipeline_updated",
                "data": data,
            }],
        }
    except Exception as e:
        log.error("PipelineDetector failed: %s", e)
        return {**state, "error": str(e)}
