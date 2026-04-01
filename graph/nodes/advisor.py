from __future__ import annotations

import json
import logging

from config import settings
from graph.nodes.context import (
    combined_fragments,
    combined_refusals,
    merged_knowledge,
    recent_probe_pairs,
    total_probe_count,
)
from graph.state import EnumerationState
from llm.client import get_client
from prompts.templates import STRATEGY_ADVISOR_SYSTEM, STRATEGY_ADVISOR_USER

log = logging.getLogger(__name__)


async def strategy_advisor(state: EnumerationState) -> EnumerationState:
    client = get_client(timeout=settings.ollama_timeout)

    fragments = []
    for fragment in combined_fragments(state):
        text = fragment.get("text", "").strip()
        if text:
            fragments.append(f"- {text[:100]}")

    triggers = []
    for refusal in combined_refusals(state):
        trigger = refusal.get("confirmed_trigger") or refusal.get("trigger_candidate", "")
        if trigger:
            triggers.append(f"- {trigger}")

    knowledge = merged_knowledge(state)
    pipeline_info = (
        json.dumps(state.get("pipeline_json", {}), indent=2)
        if state.get("pipeline_json")
        else "Not yet detected"
    )

    prompt = STRATEGY_ADVISOR_USER.format(
        operator_guidance=state.get("probe_guidance", "").strip() or "None",
        reconstructed_prompt=state.get("reconstructed_prompt", "[No fragments yet]"),
        fragment_count=len(fragments),
        fragments="\n".join(fragments) or "None",
        refusal_triggers="\n".join(triggers) or "None",
        pipeline_info=pipeline_info,
        known_tools=", ".join(knowledge.get("tools", [])) or "None",
        known_constraints=", ".join(knowledge.get("constraints", [])) or "None",
        probe_count=total_probe_count(state),
        recent_probes=recent_probe_pairs(state),
    )

    try:
        raw = await client.chat(
            system=STRATEGY_ADVISOR_SYSTEM,
            user=prompt,
            temperature=settings.suggestion_temperature,
            max_tokens=settings.max_suggestion_tokens,
        )
        data = json.loads(raw)
        if not isinstance(data, list):
            data = []
        strategies = [
            {"text": s.get("text", ""), "vector": s.get("vector", "INDIRECT"), "rationale": s.get("rationale", "")}
            for s in data[:5]
        ]
        return {
            **state,
            "strategies": strategies,
            "events": state.get("events", []) + [{
                "type": "strategies_updated",
                "data": {"strategies": strategies},
            }],
        }
    except Exception as e:
        log.error("StrategyAdvisor failed: %s", e)
        return {**state, "strategies": [], "error": str(e)}
