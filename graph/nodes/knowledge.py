from __future__ import annotations

import logging
from textwrap import shorten

from config import settings
from graph.progress import emit_agent_step
from graph.state import EnumerationState
from llm.client import get_client
from llm.json_repair import extract_json
from llm.runtime_config import runtime_cfg
from prompts.templates import (
    FRAGMENT_EXTRACTOR_SYSTEM,
    FRAGMENT_EXTRACTOR_USER,
    KNOWLEDGE_UPDATER_SYSTEM,
    KNOWLEDGE_UPDATER_USER,
)

log = logging.getLogger(__name__)


def _text_list(value: object) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            cleaned = str(item).strip()
            if cleaned:
                items.append(cleaned)
        return items
    return []


def _fallback_fragment(response_text: str) -> tuple[str, float, str]:
    text = response_text.strip()
    if not text:
        return "", 0.0, "unknown"

    try:
        parsed = extract_json(text)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        for key, confidence in (
            ("system_prompt", 0.98),
            ("developer_instructions", 0.96),
            ("instructions", 0.94),
            ("prompt", 0.9),
            ("system", 0.9),
        ):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), confidence, "beginning"

        for key in ("constraints", "rules", "safety_directives"):
            items = _text_list(parsed.get(key))
            if items:
                return "\n".join(items), 0.86, "middle"

        for key in ("purpose", "persona", "role"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), 0.74, "beginning"

        condensed: list[str] = []
        for key, value in parsed.items():
            if isinstance(value, str) and value.strip():
                condensed.append(f"{key}: {value.strip()}")
            else:
                items = _text_list(value)
                if items:
                    condensed.append(f"{key}: {'; '.join(items)}")
            if len("\n".join(condensed)) >= 1200:
                break
        if condensed:
            return "\n".join(condensed)[:1200].strip(), 0.65, "unknown"

    return text[:1600], 0.55, "unknown"


async def fragment_extractor(state: EnumerationState) -> EnumerationState:
    await emit_agent_step(
        state,
        phase="fragment",
        label="Recovering prompt fragment",
        status="start",
        detail="Scanning leak output for reusable prompt text.",
    )
    fallback_fragment, fallback_confidence, fallback_position = _fallback_fragment(state["response_text"])
    if runtime_cfg.prefers_tight_loops and fallback_fragment and fallback_confidence >= 0.9:
        await emit_agent_step(
            state,
            phase="fragment",
            label="Recovering prompt fragment",
            status="done",
            detail=shorten(fallback_fragment, width=120, placeholder="..."),
        )
        return {
            **state,
            "fragment_text": fallback_fragment,
            "fragment_confidence": fallback_confidence,
            "fragment_position": fallback_position,
            "events": state.get("events", []) + [{
                "type": "fragment_found",
                "data": {
                    "fragment": fallback_fragment,
                    "confidence": fallback_confidence,
                    "position_hint": fallback_position,
                },
            }],
        }

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
            max_tokens=runtime_cfg.analysis_token_budget(settings.max_analysis_tokens),
        )
        data = extract_json(raw)
        fragment = data.get("fragment", "").strip()
        confidence = float(data.get("confidence", 0.5))
        position_hint = data.get("position_hint", "unknown")
        if not fragment:
            fragment, confidence, position_hint = _fallback_fragment(state["response_text"])
        await emit_agent_step(
            state,
            phase="fragment",
            label="Recovering prompt fragment",
            status="done",
            detail=shorten(fragment or client.last_reasoning or "No fragment recovered.", width=120, placeholder="..."),
        )
        return {
            **state,
            "fragment_text": fragment,
            "fragment_confidence": confidence,
            "fragment_position": position_hint,
            "events": state.get("events", []) + [{
                "type": "fragment_found",
                "data": {
                    "fragment": fragment,
                    "confidence": confidence,
                    "position_hint": position_hint,
                },
            }],
        }
    except Exception as e:
        fragment, confidence, position_hint = fallback_fragment, fallback_confidence, fallback_position
        if fragment:
            log.warning("FragmentExtractor failed, using fallback: %s", e)
            await emit_agent_step(
                state,
                phase="fragment",
                label="Recovering prompt fragment",
                status="done",
                detail=shorten(fragment, width=120, placeholder="..."),
            )
            return {
                **state,
                "fragment_text": fragment,
                "fragment_confidence": confidence,
                "fragment_position": position_hint,
                "events": state.get("events", []) + [{
                    "type": "fragment_found",
                    "data": {
                        "fragment": fragment,
                        "confidence": confidence,
                        "position_hint": position_hint,
                    },
                }],
            }
        log.error("FragmentExtractor failed: %s", e)
        await emit_agent_step(
            state,
            phase="fragment",
            label="Recovering prompt fragment",
            status="error",
            detail=str(e),
        )
        return {
            **state,
            "fragment_text": "",
            "fragment_confidence": 0.0,
            "fragment_position": "unknown",
            "error": str(e),
        }


async def knowledge_updater(state: EnumerationState) -> EnumerationState:
    await emit_agent_step(
        state,
        phase="knowledge",
        label="Extracting disclosed knowledge",
        status="start",
        detail="Condensing tools, constraints, and persona clues.",
    )
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
            max_tokens=runtime_cfg.analysis_token_budget(settings.max_analysis_tokens),
        )
        data = extract_json(raw)
        detail_bits = []
        for key in ("tools", "constraints", "persona"):
            count = len(data.get(key, []) or [])
            if count:
                detail_bits.append(f"{key}={count}")
        await emit_agent_step(
            state,
            phase="knowledge",
            label="Extracting disclosed knowledge",
            status="done",
            detail=", ".join(detail_bits) or shorten(client.last_reasoning or "No structured knowledge added.", width=120, placeholder="..."),
        )
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
        await emit_agent_step(
            state,
            phase="knowledge",
            label="Extracting disclosed knowledge",
            status="error",
            detail=str(e),
        )
        return {**state, "new_knowledge": {}, "error": str(e)}
