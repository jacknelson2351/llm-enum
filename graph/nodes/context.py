from __future__ import annotations

from typing import Any

from graph.state import EnumerationState

KNOWLEDGE_KEYS = ("tools", "constraints", "persona", "raw_facts")


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def merged_knowledge(state: EnumerationState) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {key: [] for key in KNOWLEDGE_KEYS}
    for source in (state.get("session_knowledge", {}), state.get("new_knowledge", {})):
        for key in KNOWLEDGE_KEYS:
            merged[key].extend(source.get(key, []))
    return {key: _dedupe_strings(values) for key, values in merged.items()}


def combined_fragments(state: EnumerationState) -> list[dict[str, Any]]:
    fragments = [dict(fragment) for fragment in state.get("session_fragments", [])]
    current_text = state.get("fragment_text", "").strip()
    if current_text:
        current_fragment = {
            "text": current_text,
            "confidence": float(state.get("fragment_confidence", 0.0)),
            "position_hint": state.get("fragment_position", "unknown"),
        }
        if not any(
            fragment.get("text") == current_fragment["text"]
            and fragment.get("position_hint") == current_fragment["position_hint"]
            for fragment in fragments
        ):
            fragments.append(current_fragment)
    return fragments


def combined_refusals(state: EnumerationState) -> list[dict[str, Any]]:
    refusals = [dict(refusal) for refusal in state.get("session_refusals", [])]
    if state.get("classification") == "REFUSAL":
        current_refusal = {
            "probe_text": state.get("probe_text", ""),
            "refusal_type": state.get("refusal_type", "hard_refusal"),
            "trigger_candidate": state.get("trigger_candidate", ""),
            "confirmed_trigger": state.get("bisect_confirmed_trigger", ""),
            "bisect_depth": int(state.get("bisect_iteration", 0)),
        }
        if not any(
            refusal.get("probe_text") == current_refusal["probe_text"]
            and refusal.get("trigger_candidate") == current_refusal["trigger_candidate"]
            and refusal.get("confirmed_trigger") == current_refusal["confirmed_trigger"]
            for refusal in refusals
        ):
            refusals.append(current_refusal)
    return refusals


def total_probe_count(state: EnumerationState) -> int:
    count = len(state.get("session_probes", []))
    if state.get("probe_text") or state.get("response_text"):
        count += 1
    return count


def classification_count(state: EnumerationState, classification: str) -> int:
    count = sum(
        1
        for probe in state.get("session_probes", [])
        if probe.get("classification") == classification
    )
    if state.get("classification") == classification:
        count += 1
    return count


def recent_probe_pairs(state: EnumerationState, limit: int = 3) -> str:
    probes = [dict(probe) for probe in state.get("session_probes", [])]
    if state.get("probe_text") or state.get("response_text"):
        probes.append(
            {
                "probe_text": state.get("probe_text", ""),
                "response_text": state.get("response_text", ""),
                "classification": state.get("classification", "PENDING"),
            }
        )

    recent = probes[-limit:]
    if not recent:
        return "None yet"

    lines: list[str] = []
    for probe in recent:
        classification = probe.get("classification", "UNKNOWN")
        lines.append(
            f"[{classification}] PROBE: {(probe.get('probe_text') or '')[:200]}\n"
            f"RESPONSE: {(probe.get('response_text') or '')[:200]}"
        )
    return "\n\n".join(lines)
