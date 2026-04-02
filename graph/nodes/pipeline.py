from __future__ import annotations

from collections import Counter, defaultdict
import json
import logging
import re
from typing import Any

from config import settings
from graph.nodes.context import (
    classification_count,
    combined_fragments,
    combined_refusals,
    merged_knowledge,
    recent_probe_pairs,
    total_probe_count,
)
from graph.session_insights import classify_guardrail_family, classify_surface, probe_signature
from graph.state import EnumerationState
from llm.client import get_client
from llm.json_repair import extract_json
from prompts.templates import PIPELINE_DETECTOR_SYSTEM, PIPELINE_DETECTOR_USER

log = logging.getLogger(__name__)

_PROMPT_PLACEHOLDER = "[No fragments discovered yet]"
_AUTO_NODE_PREFIX = "obs_"
_STYLE_LABELS = {
    "structured": "structured",
    "tooling": "tool-aware",
    "listed": "list-style",
    "guarded": "guarded",
    "longform": "long-form",
    "conversational": "conversational",
    "plain": "plain",
}
_KIND_LABELS = {
    "leak": "Leak path",
    "guarded": "Guarded path",
    "tool": "Tool path",
    "general": "General path",
}
_KIND_ORDER = {"leak": 0, "guarded": 1, "tool": 2, "general": 3}


def _node_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "node"


def _enum_value(value: Any) -> str:
    return getattr(value, "value", value) or ""


def _normalize_pipeline(data: object) -> dict:
    if not isinstance(data, dict):
        return {
            "nodes": [],
            "edges": [],
            "overall_confidence": 0.0,
            "topology_type": "unknown",
        }
    return {
        "nodes": [dict(node) for node in list(data.get("nodes", []) or [])],
        "edges": [dict(edge) for edge in list(data.get("edges", []) or [])],
        "overall_confidence": float(data.get("overall_confidence", 0.0) or 0.0),
        "topology_type": data.get("topology_type", "unknown") or "unknown",
    }


def _upsert_node(nodes: list[dict], node: dict) -> str:
    node_id = str(node.get("id") or "")
    if not node_id:
        raise ValueError("Pipeline node id is required")

    for existing in nodes:
        if existing.get("id") != node_id:
            continue
        existing["confidence"] = max(
            float(existing.get("confidence", 0.0) or 0.0),
            float(node.get("confidence", 0.0) or 0.0),
        )
        for key in ("label", "suggested_strategy", "summary", "group", "sprite"):
            if not existing.get(key) and node.get(key):
                existing[key] = node.get(key)
        merged_evidence = list(existing.get("evidence", []) or [])
        for item in node.get("evidence", []) or []:
            if item and item not in merged_evidence:
                merged_evidence.append(item)
        existing["evidence"] = merged_evidence
        return node_id

    nodes.append(node)
    return node_id


def _edge_exists(edges: list[dict], from_id: str, to_id: str) -> bool:
    return any(
        edge.get("from_id") == from_id and edge.get("to_id") == to_id
        for edge in edges
    )


def _connect(edges: list[dict], from_id: str, to_id: str, label: str | None = None) -> None:
    if not from_id or not to_id or _edge_exists(edges, from_id, to_id):
        return
    edges.append({"from_id": from_id, "to_id": to_id, "label": label})


def _tool_label(tool_name: str) -> str:
    cleaned = tool_name.strip()
    if not cleaned:
        return "Tool"
    return cleaned.replace("_", " ")


def _group_for_type(node_type: str) -> str:
    if node_type == "prompt_surface":
        return "prompt"
    if node_type in {"guard_pre", "guard_post", "router", "orchestrator"}:
        return "control"
    if node_type == "retriever":
        return "context"
    if node_type == "tool_executor":
        return "tooling"
    return "response"


def _sprite_for_type(node_type: str) -> str:
    return {
        "prompt_surface": "crystal",
        "guard_pre": "shield",
        "guard_post": "shield",
        "router": "splitter",
        "orchestrator": "hub",
        "worker_llm": "prism",
        "retriever": "kite",
        "tool_executor": "cube",
    }.get(node_type, "node")


def _truncate(text: str, width: int = 54) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= width:
        return compact
    return compact[: width - 1].rstrip() + "…"


def _annotate_existing_nodes(nodes: list[dict]) -> None:
    for node in nodes:
        node_type = str(node.get("type") or "unknown")
        node.setdefault("group", _group_for_type(node_type))
        node.setdefault("sprite", _sprite_for_type(node_type))
        if not node.get("summary"):
            evidence = list(node.get("evidence", []) or [])
            node["summary"] = _truncate(evidence[0], width=48) if evidence else ""


def _prune_generated_nodes(pipeline: dict) -> dict:
    removable_ids: set[str] = set()
    kept_nodes: list[dict] = []

    for node in pipeline["nodes"]:
        node_id = str(node.get("id") or "")
        label = str(node.get("label") or "").strip().lower()
        node_type = str(node.get("type") or "")
        if (
            node_id.startswith(_AUTO_NODE_PREFIX)
            or node_id == "responder_llm"
            or node_id.startswith("tool_")
            or (node_type == "worker_llm" and label in {"responder llm", "worker llm", "main llm"})
        ):
            removable_ids.add(node_id)
            continue
        kept_nodes.append(node)

    kept_edges = [
        edge
        for edge in pipeline["edges"]
        if edge.get("from_id") not in removable_ids and edge.get("to_id") not in removable_ids
    ]
    pipeline["nodes"] = kept_nodes
    pipeline["edges"] = kept_edges
    return pipeline


def _all_probe_records(state: EnumerationState) -> list[dict[str, Any]]:
    probes = [dict(probe) for probe in state.get("session_probes", [])]
    if state.get("probe_text") or state.get("response_text"):
        probes.append(
            {
                "probe_text": state.get("probe_text", ""),
                "response_text": state.get("response_text", ""),
                "classification": state.get("classification", "PENDING"),
            }
        )
    return probes


def _response_style_tag(response_text: str, classification: str) -> str:
    lowered = (response_text or "").lower()
    stripped = response_text.strip()
    if not stripped:
        return "plain"
    if "```" in response_text or re.match(r"^\s*[\[{]", response_text):
        return "structured"
    if classification == "TOOL_DISCLOSURE" or any(
        marker in lowered for marker in ("tool", "browser", "function", "api", "plugin")
    ):
        return "tooling"
    if re.search(r"^\s*[-*]\s+\w", response_text, re.MULTILINE) or re.search(
        r"^\s*\d+\.\s+\w", response_text, re.MULTILINE
    ):
        return "listed"
    if classification == "REFUSAL" or any(
        phrase in lowered
        for phrase in ("i'm sorry", "i am sorry", "i cannot", "i can't", "i can’t", "unable to")
    ):
        return "guarded"
    if len(stripped.split()) > 140:
        return "longform"
    if re.search(r"\b(i|we)\b", lowered):
        return "conversational"
    return "plain"


def _channel_kind(record: dict[str, Any]) -> str:
    classification = _enum_value(record.get("classification", "NEUTRAL")) or "NEUTRAL"
    if classification == "LEAK":
        return "leak"
    if classification == "REFUSAL":
        return "guarded"
    if classification == "TOOL_DISCLOSURE":
        return "tool"
    return "general"


def _format_counts(counter: Counter[str], *, lower: bool = True, limit: int = 3) -> str:
    parts: list[str] = []
    for key, value in counter.most_common(limit):
        if not key or value <= 0:
            continue
        label = key.lower() if lower else key
        parts.append(f"{label}={value}")
    return ", ".join(parts) or "none"


def _response_channels(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clusters: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        classification = _enum_value(record.get("classification", "NEUTRAL")) or "NEUTRAL"
        if classification == "ERROR":
            continue
        probe_text = str(record.get("probe_text", "") or "")
        response_text = str(record.get("response_text", "") or "")
        kind = _channel_kind(record)
        style = _response_style_tag(response_text, classification)
        key = (kind, style)
        row = clusters.setdefault(
            key,
            {
                "kind": kind,
                "style": style,
                "count": 0,
                "surfaces": Counter(),
                "classifications": Counter(),
                "examples": [],
            },
        )
        row["count"] += 1
        row["surfaces"][classify_surface(probe_text)] += 1
        row["classifications"][classification] += 1
        example = probe_signature(response_text or probe_text)
        if example and example not in row["examples"] and len(row["examples"]) < 2:
            row["examples"].append(example)

    ordered = sorted(
        clusters.values(),
        key=lambda row: (
            _KIND_ORDER.get(row["kind"], 99),
            -row["count"],
            row["style"],
        ),
    )
    kind_counts = Counter(row["kind"] for row in ordered)
    channels: list[dict[str, Any]] = []
    for row in ordered:
        kind = row["kind"]
        style = row["style"]
        label = _KIND_LABELS[kind]
        if kind_counts[kind] > 1:
            label = f"{label} · {_STYLE_LABELS.get(style, style)}"
        summary = f"{row['count']} response(s) · {_STYLE_LABELS.get(style, style)}"
        evidence = [
            f"Observed {row['count']} {_KIND_LABELS[kind].lower()} output(s)",
            f"Surfaces hit: {_format_counts(row['surfaces'])}",
        ]
        if row["examples"]:
            evidence.append(f"Response signature: {' | '.join(row['examples'])}")
        confidence = {
            "leak": 0.91,
            "tool": 0.86,
            "guarded": 0.8,
            "general": 0.68,
        }.get(kind, 0.6)
        confidence = min(0.97, confidence + min(0.08, (row["count"] - 1) * 0.03))
        channels.append(
            {
                "id": f"{_AUTO_NODE_PREFIX}channel_{kind}_{style}",
                "type": "worker_llm",
                "label": label,
                "confidence": confidence,
                "evidence": evidence,
                "suggested_strategy": "Differentiate this response path with more targeted probes.",
                "summary": summary,
                "group": "response",
                "sprite": {
                    "leak": "shard",
                    "tool": "prism-tool",
                    "guarded": "prism-guarded",
                    "general": "prism",
                }.get(kind, "prism"),
                "_kind": kind,
                "_count": row["count"],
            }
        )
    return channels


def _prompt_surface_node(state: EnumerationState) -> dict[str, Any] | None:
    fragments = combined_fragments(state)
    reconstructed = (state.get("reconstructed_prompt") or "").strip()
    if not fragments and (not reconstructed or reconstructed == _PROMPT_PLACEHOLDER):
        return None

    evidence: list[str] = []
    if fragments:
        evidence.append(f"Recovered {len(fragments)} fragment(s) from leaked output")
    if reconstructed and reconstructed != _PROMPT_PLACEHOLDER:
        evidence.append("Reconstructed prompt text is available for this session")
    return {
        "id": f"{_AUTO_NODE_PREFIX}prompt_surface",
        "type": "prompt_surface",
        "label": "Prompt surface",
        "confidence": 0.9 if fragments else 0.72,
        "evidence": evidence,
        "suggested_strategy": "Use recovered prompt text to target exact boundaries and gaps.",
        "summary": f"{len(fragments)} fragment(s) recovered" if fragments else "Reconstructed prompt available",
        "group": "prompt",
        "sprite": "crystal",
    }


def _guardrail_node(state: EnumerationState) -> dict[str, Any] | None:
    refusals = combined_refusals(state)
    if not refusals:
        return None

    families: Counter[str] = Counter()
    for refusal in refusals:
        family = classify_guardrail_family(
            str(refusal.get("probe_text", "") or ""),
            trigger_text=str(
                refusal.get("confirmed_trigger", "")
                or refusal.get("trigger_candidate", "")
                or ""
            ),
        )
        families[family] += 1

    top_families = [(family, count) for family, count in families.most_common(2) if family]
    if not top_families:
        top_families = [("UNKNOWN", len(refusals))]

    summary = " · ".join(
        family.replace("_", " ").lower() for family, _ in top_families
    )
    evidence = [
        f"Observed {len(refusals)} refusal event(s)",
        "Guardrail families: "
        + ", ".join(f"{family} x{count}" for family, count in top_families),
    ]
    return {
        "id": f"{_AUTO_NODE_PREFIX}guardrail_gate",
        "type": "guard_pre",
        "label": "Guardrail gate",
        "confidence": min(0.94, 0.74 + len(refusals) * 0.04),
        "evidence": evidence,
        "suggested_strategy": "Target the strongest refusal family with narrow variants.",
        "summary": summary,
        "group": "control",
        "sprite": "shield",
    }


def _router_summary(channels: list[dict[str, Any]]) -> tuple[str, list[str]]:
    kind_counts = Counter(channel["_kind"] for channel in channels)
    evidence = [
        "Observed behavior split: "
        + ", ".join(
            f"{kind}={kind_counts.get(kind, 0)}"
            for kind in ("leak", "guarded", "tool", "general")
            if kind_counts.get(kind, 0)
        )
    ]
    return f"{len(channels)} response path(s)", evidence


def _materialize_pipeline(data: object, state: EnumerationState) -> dict:
    pipeline = _prune_generated_nodes(_normalize_pipeline(data))
    nodes = pipeline["nodes"]
    edges = pipeline["edges"]
    _annotate_existing_nodes(nodes)

    knowledge = merged_knowledge(state)
    probe_records = _all_probe_records(state)
    prompt_node = _prompt_surface_node(state)
    if prompt_node:
        _upsert_node(nodes, prompt_node)

    guard_node = _guardrail_node(state)
    guard_id = ""
    if guard_node:
        guard_id = _upsert_node(nodes, guard_node)

    channels = _response_channels(probe_records)
    channel_ids: list[str] = []
    for channel in channels:
        channel_ids.append(_upsert_node(nodes, {k: v for k, v in channel.items() if not k.startswith("_")}))

    router_id = ""
    existing_router = next(
        (
            node
            for node in nodes
            if str(node.get("type") or "") in {"router", "orchestrator"}
        ),
        None,
    )
    if existing_router and len(channel_ids) > 1:
        router_id = str(existing_router.get("id") or "")
        summary, evidence = _router_summary(channels)
        _upsert_node(
            nodes,
            {
                "id": router_id,
                "type": existing_router.get("type", "router"),
                "label": existing_router.get("label") or "Behavior router",
                "confidence": 0.84,
                "evidence": evidence,
                "suggested_strategy": existing_router.get("suggested_strategy", ""),
                "summary": existing_router.get("summary") or summary,
                "group": "control",
                "sprite": "splitter",
            },
        )
    elif len(channel_ids) > 1:
        summary, evidence = _router_summary(channels)
        router_id = _upsert_node(
            nodes,
            {
                "id": f"{_AUTO_NODE_PREFIX}behavior_router",
                "type": "router",
                "label": "Behavior router",
                "confidence": 0.84,
                "evidence": evidence,
                "suggested_strategy": "Drive probes through each observed response path separately.",
                "summary": summary,
                "group": "control",
                "sprite": "splitter",
            },
        )

    if guard_id and router_id:
        _connect(edges, guard_id, router_id, "policy")

    channel_kind_lookup = {channel["id"]: channel["_kind"] for channel in channels}
    for channel_id in channel_ids:
        kind = channel_kind_lookup.get(channel_id, "general")
        if router_id:
            _connect(edges, router_id, channel_id, kind)
        elif guard_id:
            _connect(edges, guard_id, channel_id, kind)

    if prompt_node:
        prompt_id = prompt_node["id"]
        if guard_id:
            _connect(edges, prompt_id, guard_id, "instructions")
        elif router_id:
            _connect(edges, prompt_id, router_id, "instructions")
        else:
            for channel_id in channel_ids:
                _connect(
                    edges,
                    prompt_id,
                    channel_id,
                    "instructions" if len(channel_ids) == 1 else "conditions",
                )

    tool_names: list[str] = []
    for tool_name in knowledge.get("tools", []):
        cleaned = str(tool_name).strip()
        if cleaned and cleaned not in tool_names:
            tool_names.append(cleaned)

    tool_anchor_candidates = [
        channel_id
        for channel_id in channel_ids
        if channel_kind_lookup.get(channel_id) in {"tool", "general"}
    ]
    tool_anchor = tool_anchor_candidates[0] if tool_anchor_candidates else (router_id or guard_id)
    for tool_name in tool_names:
        tool_id = f"tool_{_node_slug(tool_name)}"
        _upsert_node(
            nodes,
            {
                "id": tool_id,
                "type": "tool_executor",
                "label": _tool_label(tool_name),
                "confidence": 0.84,
                "evidence": [f"Tool disclosed in session evidence: {tool_name}"],
                "suggested_strategy": f"Probe how {tool_name} is exposed and constrained.",
                "summary": "disclosed tool",
                "group": "tooling",
                "sprite": "cube",
            },
        )
        if tool_anchor:
            _connect(edges, tool_anchor, tool_id, "tool call")

    response_count = len(channel_ids)
    tool_count = len(tool_names)
    if response_count > 1:
        pipeline["topology_type"] = "parallel"
    elif tool_count:
        pipeline["topology_type"] = "hub-spoke"
    elif guard_id and response_count:
        pipeline["topology_type"] = "sequential"
    elif response_count:
        pipeline["topology_type"] = "linear"
    elif nodes:
        pipeline["topology_type"] = pipeline["topology_type"] or "unknown"

    synthesized_confidence = 0.34 + min(0.3, len(probe_records) * 0.05)
    if prompt_node:
        synthesized_confidence += 0.14
    if guard_id:
        synthesized_confidence += 0.08
    if router_id:
        synthesized_confidence += 0.08
    if tool_count:
        synthesized_confidence += 0.08
    synthesized_confidence = min(0.97, synthesized_confidence)
    pipeline["overall_confidence"] = max(
        float(pipeline.get("overall_confidence", 0.0) or 0.0),
        synthesized_confidence,
    )

    return pipeline


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
        data = _materialize_pipeline(extract_json(raw), state)
        return {
            **state,
            "pipeline_json": data,
            "events": state.get("events", []) + [{
                "type": "pipeline_updated",
                "data": data,
            }],
        }
    except Exception as e:
        data = _materialize_pipeline(state.get("pipeline_json", {}), state)
        if data.get("nodes"):
            log.warning("PipelineDetector failed, using synthesized topology: %s", e)
            return {
                **state,
                "pipeline_json": data,
                "error": str(e),
                "events": state.get("events", []) + [{
                    "type": "pipeline_updated",
                    "data": data,
                }],
            }
        log.error("PipelineDetector failed: %s", e)
        return {**state, "error": str(e)}
