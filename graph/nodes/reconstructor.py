from __future__ import annotations

import logging
from textwrap import shorten

from config import settings
from graph.progress import emit_agent_step
from graph.nodes.context import combined_fragments, merged_knowledge
from graph.state import EnumerationState
from llm.client import get_client
from llm.runtime_config import runtime_cfg
from prompts.templates import PROMPT_RECONSTRUCTOR_SYSTEM, PROMPT_RECONSTRUCTOR_USER

log = logging.getLogger(__name__)


_POSITION_ORDER = {
    "beginning": 0,
    "middle": 1,
    "end": 2,
    "unknown": 3,
}


def _ordered_fragments(fragments: list[dict]) -> list[dict]:
    return sorted(
        fragments,
        key=lambda fragment: (
            _POSITION_ORDER.get(fragment.get("position_hint", "unknown"), 3),
            -float(fragment.get("confidence", 0.0)),
            fragment.get("text", ""),
        ),
    )


def _fallback_reconstruction(fragments: list[dict]) -> str:
    ordered = _ordered_fragments(fragments)
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
    await emit_agent_step(
        state,
        phase="reconstruct",
        label="Rebuilding prompt",
        status="start",
        detail="Stitching recovered prompt fragments into a usable prompt view.",
    )
    fragment_records = combined_fragments(state)
    fragments = []
    ordered_fragments = _ordered_fragments(fragment_records)
    for fragment in ordered_fragments[:8]:
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
        await emit_agent_step(
            state,
            phase="reconstruct",
            label="Rebuilding prompt",
            status="done",
            detail="No fragments yet.",
        )
        return {
            **state,
            "reconstructed_prompt": existing_prompt or "[No fragments discovered yet]",
        }

    fallback_prompt = _fallback_reconstruction(fragment_records)
    if runtime_cfg.prefers_tight_loops and (
        len(fragment_records) > 5 or len("\n".join(fragments)) > 1400
    ):
        await emit_agent_step(
            state,
            phase="reconstruct",
            label="Rebuilding prompt",
            status="done",
            detail=shorten(fallback_prompt, width=120, placeholder="..."),
        )
        return {
            **state,
            "reconstructed_prompt": fallback_prompt,
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
            max_tokens=runtime_cfg.reconstruction_token_budget(settings.max_suggestion_tokens),
        )
        await emit_agent_step(
            state,
            phase="reconstruct",
            label="Rebuilding prompt",
            status="done",
            detail=shorten(raw.strip() or client.last_reasoning or "Prompt stitched.", width=120, placeholder="..."),
        )
        return {**state, "reconstructed_prompt": raw.strip()}
    except Exception as e:
        log.warning("PromptReconstructor failed, using fallback: %s", e)
        await emit_agent_step(
            state,
            phase="reconstruct",
            label="Rebuilding prompt",
            status="done",
            detail=shorten(fallback_prompt, width=120, placeholder="..."),
        )
        return {
            **state,
            "reconstructed_prompt": fallback_prompt,
        }
