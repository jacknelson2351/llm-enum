from __future__ import annotations

import logging

from config import settings
from graph.nodes.context import combined_fragments, merged_knowledge
from graph.state import EnumerationState
from llm.client import get_client
from prompts.templates import PROMPT_RECONSTRUCTOR_SYSTEM, PROMPT_RECONSTRUCTOR_USER

log = logging.getLogger(__name__)


async def prompt_reconstructor(state: EnumerationState) -> EnumerationState:
    fragments = []
    for fragment in combined_fragments(state):
        text = fragment.get("text", "").strip()
        if not text:
            continue
        fragments.append(
            f"[{fragment.get('position_hint', 'unknown')}] "
            f"(confidence: {float(fragment.get('confidence', 0.0)):.0%}) "
            f"{text}"
        )

    if not fragments:
        existing_prompt = state.get("reconstructed_prompt", "").strip()
        return {
            **state,
            "reconstructed_prompt": existing_prompt or "[No fragments discovered yet]",
        }

    knowledge = merged_knowledge(state)
    prompt = PROMPT_RECONSTRUCTOR_USER.format(
        fragments="\n".join(fragments),
        constraints=", ".join(knowledge.get("constraints", [])) or "None known",
        persona=", ".join(knowledge.get("persona", [])) or "None known",
    )

    client = get_client(timeout=settings.ollama_timeout)
    try:
        raw = await client.chat(
            system=PROMPT_RECONSTRUCTOR_SYSTEM,
            user=prompt,
            temperature=settings.analysis_temperature,
            max_tokens=settings.max_suggestion_tokens,
        )
        return {**state, "reconstructed_prompt": raw.strip()}
    except Exception as e:
        log.error("PromptReconstructor failed: %s", e)
        return {**state, "error": str(e)}
