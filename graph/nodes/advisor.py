from __future__ import annotations

import json
import logging

from config import settings
from graph.nodes.context import (
    combined_fragments,
    combined_refusals,
    gap_analysis_digest,
    guardrail_hypothesis_digest,
    merged_knowledge,
    probe_history_digest,
    recent_probe_pairs,
    refusal_cluster_digest,
    surface_coverage_digest,
    total_probe_count,
)
from graph.session_insights import rerank_strategies
from graph.state import EnumerationState
from llm.client import get_client
from llm.json_repair import extract_json
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

    reconstructed = state.get("reconstructed_prompt", "[No fragments yet]")
    # Trim reconstructed prompt for 20B context budget
    if len(reconstructed) > 300:
        reconstructed = reconstructed[:300] + "..."

    prompt = STRATEGY_ADVISOR_USER.format(
        gap_analysis=gap_analysis_digest(state),
        operator_guidance=state.get("probe_guidance", "").strip() or "None",
        reconstructed_prompt=reconstructed,
        fragment_count=len(fragments),
        fragments="\n".join(fragments) or "None",
        refusal_triggers="\n".join(triggers) or "None",
        refusal_clusters=refusal_cluster_digest(state),
        guardrail_hypotheses=guardrail_hypothesis_digest(state),
        known_tools=", ".join(knowledge.get("tools", [])) or "None",
        known_constraints=", ".join(knowledge.get("constraints", [])) or "None",
        probe_history=probe_history_digest(state),
        surface_coverage=surface_coverage_digest(state),
        recent_probes=recent_probe_pairs(state),
    )

    try:
        raw = await client.chat(
            system=STRATEGY_ADVISOR_SYSTEM,
            user=prompt,
            temperature=settings.suggestion_temperature,
            max_tokens=settings.max_suggestion_tokens,
        )
        data = extract_json(raw)
        if not isinstance(data, list):
            data = []
        history = [dict(probe) for probe in state.get("session_probes", [])]
        if state.get("probe_text") or state.get("response_text"):
            history.append(
                {
                    "probe_text": state.get("probe_text", ""),
                    "response_text": state.get("response_text", ""),
                    "classification": state.get("classification", "PENDING"),
                }
            )
        strategies = rerank_strategies(data[:10], history, limit=5)
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
