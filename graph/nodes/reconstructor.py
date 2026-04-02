from __future__ import annotations

import logging

from config import settings
from graph.nodes.context import combined_fragments, merged_knowledge
from graph.state import EnumerationState
from llm.client import get_client
from prompts.templates import PROMPT_RECONSTRUCTOR_SYSTEM, PROMPT_RECONSTRUCTOR_USER

log = logging.getLogger(__name__)


_POSITION_ORDER = {
    "beginning": 0,
    "middle": 1,
    "end": 2,
    "unknown": 3,
}


def _fallback_reconstruction(fragments: list[dict]) -> str:
    ordered = sorted(
        fragments,
        key=lambda fragment: (
            _POSITION_ORDER.get(fragment.get("position_hint", "unknown"), 3),
            -float(fragment.get("confidence", 0.0)),
            fragment.get("text", ""),
        ),
    )
    parts: list[str] = []
    for fragment in ordered:
        text = fragment.get("text", "").strip()
        if text and text not in parts:
            parts.append(text)
    if not parts:
        return "[No fragments discovered yet]"
    if len(parts) == 1:
        return parts[0]
    return "\n[UNKNOWN]\n".join(parts)


async def prompt_reconstructor(state: EnumerationState) -> EnumerationState:
    fragment_records = combined_fragments(state)
    fragments = []
    for fragment in fragment_records:
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
        log.warning("PromptReconstructor failed, using fallback: %s", e)
        return {
            **state,
            "reconstructed_prompt": _fallback_reconstruction(fragment_records),
        }
