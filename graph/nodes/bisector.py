from __future__ import annotations

import json
import logging

from config import settings
from graph.state import EnumerationState
from llm.client import get_client
from prompts.templates import BISECTOR_SYSTEM, BISECTOR_USER

log = logging.getLogger(__name__)


async def refusal_bisector(state: EnumerationState) -> EnumerationState:
    trigger = state.get("trigger_candidate", "")
    if not trigger:
        return state

    words = trigger.split()
    if len(words) < settings.bisect_min_words:
        return {
            **state,
            "bisect_confirmed_trigger": trigger,
            "bisect_iteration": 0,
        }

    client = get_client(timeout=settings.ollama_timeout)
    original_probe = state["probe_text"]
    candidate = trigger
    iteration = 0
    events = list(state.get("events", []))

    try:
        while iteration < settings.bisect_max_iterations:
            words = candidate.split()
            if len(words) < settings.bisect_min_words:
                break

            mid = len(words) // 2
            left_half = " ".join(words[:mid])
            right_half = " ".join(words[mid:])

            # Test left half
            raw = await client.chat(
                system=BISECTOR_SYSTEM,
                user=BISECTOR_USER.format(
                    original_probe=original_probe,
                    candidate=left_half,
                ),
                temperature=settings.analysis_temperature,
                max_tokens=128,
            )
            left_data = json.loads(raw)

            # Test right half
            raw = await client.chat(
                system=BISECTOR_SYSTEM,
                user=BISECTOR_USER.format(
                    original_probe=original_probe,
                    candidate=right_half,
                ),
                temperature=settings.analysis_temperature,
                max_tokens=128,
            )
            right_data = json.loads(raw)

            left_trigger = left_data.get("likely_trigger", False)
            right_trigger = right_data.get("likely_trigger", False)

            iteration += 1
            if left_trigger and not right_trigger:
                candidate = left_half
            elif right_trigger and not left_trigger:
                candidate = right_half
            elif left_trigger and right_trigger:
                # Both halves trigger — pick the shorter one
                candidate = left_half if len(left_half) <= len(right_half) else right_half
            else:
                # Neither half triggers alone — the combination is needed
                break

            events.append({
                "type": "bisect_update",
                "data": {"iteration": iteration, "candidate": candidate},
            })

    except Exception as e:
        log.error("RefusalBisector failed at iteration %d: %s", iteration, e)

    return {
        **state,
        "bisect_confirmed_trigger": candidate,
        "bisect_iteration": iteration,
        "events": events,
    }
