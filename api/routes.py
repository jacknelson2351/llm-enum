from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime, timezone
import re
from textwrap import shorten

from fastapi import APIRouter, HTTPException, WebSocket
from pydantic import BaseModel

from api.ws import push_events, ws_connect
from config import LLMBackend, settings
from db.store import store
from graph.graph import graph
from graph.nodes.advisor import strategy_advisor
from graph.state import EnumerationState
from knowledge import public_reference_payload, render_reference_context
from llm.client import LLMClient
from llm.runtime_config import runtime_cfg
from models.enums import Classification, RefusalType
from models.fragments import Fragment
from models.pipeline import DetectedNode, PipelineEdge, PipelineTopology
from models.refusal import RefusalEvent
from models.session import ChatTurn, ProbeRecord, SessionState
from models.strategy import Strategy
from prompts.templates import ASSISTANT_CHAT_SYSTEM, ASSISTANT_CHAT_USER

router = APIRouter()


# ── Request/Response schemas ──

class CreateSessionReq(BaseModel):
    name: str = "Untitled Session"

class UpdateSessionReq(BaseModel):
    name: str | None = None
    probe_guidance: str | None = None

class ProbeReq(BaseModel):
    probe_text: str
    response_text: str

class LLMConfigReq(BaseModel):
    backend: str | None = None
    url: str | None = None
    model: str | None = None


class AssistantChatReq(BaseModel):
    message: str


def _pipeline_payload(session: SessionState) -> dict:
    if session.pipeline.nodes or session.pipeline.edges:
        return session.pipeline.model_dump(mode="json")
    return {}


def _build_state_from_session(
    session: SessionState,
    *,
    probe_text: str = "",
    response_text: str = "",
    probe_id: str = "",
) -> EnumerationState:
    return {
        "session_id": session.id,
        "probe_text": probe_text,
        "response_text": response_text,
        "probe_id": probe_id,
        "session_probes": [p.model_dump(mode="json") for p in session.probes],
        "session_fragments": [f.model_dump(mode="json") for f in session.fragments],
        "session_refusals": [r.model_dump(mode="json") for r in session.refusals],
        "session_knowledge": {
            key: list(session.knowledge.get(key, []))
            for key in ("tools", "constraints", "persona", "raw_facts")
        },
        "pipeline_json": _pipeline_payload(session),
        "reconstructed_prompt": session.reconstructed_prompt,
        "probe_guidance": session.probe_guidance,
        "events": [],
    }


def _update_pipeline(session: SessionState, pipeline_json: dict) -> None:
    nodes = [DetectedNode(**n) for n in pipeline_json.get("nodes", [])]
    edges = [PipelineEdge(**e) for e in pipeline_json.get("edges", [])]
    session.pipeline = PipelineTopology(
        nodes=nodes,
        edges=edges,
        overall_confidence=pipeline_json.get("overall_confidence", 0.0),
        topology_type=pipeline_json.get("topology_type", "unknown"),
        updated_at=datetime.now(timezone.utc),
    )


def _update_strategies(session: SessionState, strategies: list[dict]) -> None:
    session.strategies = [Strategy(**s) for s in strategies]


_PROBE_STOPWORDS = {
    "about", "after", "again", "also", "and", "any", "are", "assistant", "been",
    "between", "could", "from", "give", "have", "hidden", "into", "just", "like",
    "list", "make", "model", "more", "output", "prompt", "repeat", "reveal",
    "role", "should", "show", "system", "that", "them", "then", "there", "these",
    "this", "using", "what", "with", "would", "your",
}


def _probe_terms(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9_+-]+", text.lower())
        if len(token) > 2 and token not in _PROBE_STOPWORDS
    ]


def _probe_signature(text: str) -> str:
    terms = _probe_terms(text)
    if not terms:
        return shorten(" ".join(text.split()), width=72, placeholder="...") or "empty"
    return " ".join(terms[:6])


def _probe_history_digest(session: SessionState, limit: int = 8) -> str:
    if not session.probes:
        return "None"

    class_counts = Counter(probe.classification.value for probe in session.probes)
    signature_rows: dict[str, dict] = {}

    for probe in session.probes:
        signature = _probe_signature(probe.probe_text)
        row = signature_rows.setdefault(
            signature,
            {
                "count": 0,
                "classification": probe.classification.value,
                "text": shorten(probe.probe_text, width=92, placeholder="..."),
            },
        )
        row["count"] += 1
        row["classification"] = probe.classification.value

    ordered_signatures = list(signature_rows.items())[-limit:]

    lines = [
        (
            "Coverage: "
            f"total={len(session.probes)} "
            f"leak={class_counts.get('LEAK', 0)} "
            f"refusal={class_counts.get('REFUSAL', 0)} "
            f"tool={class_counts.get('TOOL_DISCLOSURE', 0)} "
            f"neutral={class_counts.get('NEUTRAL', 0)} "
            f"error={class_counts.get('ERROR', 0)}"
        ),
        "Recent unique probe signatures:",
    ]

    for _, row in ordered_signatures:
        repeat = f" x{row['count']}" if row["count"] > 1 else ""
        lines.append(
            f"- [{row['classification']}{repeat}] {row['text']}"
        )
    return "\n".join(lines)


def _format_project_summary(session: SessionState) -> str:
    knowledge = session.knowledge or {}
    pipeline = session.pipeline
    latest = session.probes[-1] if session.probes else None
    reconstructed = shorten(session.reconstructed_prompt or "None", width=220, placeholder="...")
    lines = [
        f"Project: {session.name}",
        f"Goal: {session.probe_guidance or 'None'}",
        f"Reconstructed prompt: {reconstructed}",
        (
            "Latest result: "
            f"{latest.classification.value} / {(latest.confidence * 100):.0f}% / "
            f"{shorten(latest.reasoning or 'No reasoning', width=96, placeholder='...')}"
            if latest
            else "Latest result: None"
        ),
        f"Fragments: {len(session.fragments)}",
        f"Refusals: {len(session.refusals)}",
        (
            "Pipeline: "
            f"{pipeline.topology_type} / {(pipeline.overall_confidence * 100):.0f}% / "
            f"nodes={', '.join(node.label for node in pipeline.nodes) if pipeline.nodes else 'none'}"
        ),
        f"Known tools: {', '.join(knowledge.get('tools', [])[:6]) or 'None'}",
        f"Known constraints: {', '.join(knowledge.get('constraints', [])[:6]) or 'None'}",
        f"Known persona: {', '.join(knowledge.get('persona', [])[:4]) or 'None'}",
        "Tried probes digest:",
        _probe_history_digest(session),
    ]

    if session.refusals:
        lines.append("Refusal triggers:")
        for refusal in session.refusals[-3:]:
            trigger = refusal.confirmed_trigger or refusal.trigger_candidate or "unknown"
            lines.append(
                f"- {refusal.refusal_type.value}: {shorten(trigger, width=72, placeholder='...')}"
            )

    if session.strategies:
        lines.append("Current suggestions:")
        for strategy in session.strategies[:3]:
            lines.append(
                f"- {strategy.vector}: {shorten(strategy.text, width=96, placeholder='...')}"
            )

    return "\n".join(lines)


def _format_chat_history(session: SessionState, limit: int = 6) -> str:
    if not session.assistant_chat:
        return "None"
    return "\n".join(
        f"{turn.role.upper()}: {shorten(turn.content, width=180, placeholder='...')}"
        for turn in session.assistant_chat[-limit:]
    )


def _reference_focus_text(session: SessionState, question: str) -> str:
    latest = session.probes[-1] if session.probes else None
    parts = [
        question,
        session.probe_guidance,
        session.reconstructed_prompt,
        latest.probe_text if latest else "",
        latest.response_text if latest else "",
        " ".join(_probe_signature(probe.probe_text) for probe in session.probes[-8:]),
        " ".join(session.knowledge.get("tools", [])[:6]),
        " ".join(session.knowledge.get("constraints", [])[:6]),
        " ".join(
            (
                refusal.confirmed_trigger or refusal.trigger_candidate or ""
                for refusal in session.refusals[-3:]
            )
        ),
    ]
    return " ".join(part for part in parts if part)


# ── Session endpoints ──

@router.post("/sessions")
async def create_session(req: CreateSessionReq):
    session = await store.create_session(req.name)
    return {"id": session.id, "name": session.name, "created_at": session.created_at.isoformat()}

@router.get("/sessions")
async def list_sessions():
    return await store.list_sessions()

@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")
    return session.model_dump(mode="json")

@router.patch("/sessions/{session_id}")
async def update_session(session_id: str, req: UpdateSessionReq):
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")
    if req.name is None and req.probe_guidance is None:
        raise HTTPException(400, detail="No session fields provided")
    if req.name is not None:
        session.name = req.name.strip() or "Untitled Session"
    if req.probe_guidance is not None:
        session.probe_guidance = req.probe_guidance.strip()
    await store.update_session_state(session)
    return session.model_dump(mode="json")

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    ok = await store.delete_session(session_id)
    if not ok:
        raise HTTPException(404, detail="Session not found")
    return {"ok": True}

@router.get("/sessions/{session_id}/fragments")
async def get_fragments(session_id: str):
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")
    return [f.model_dump(mode="json") for f in session.fragments]

@router.get("/sessions/{session_id}/refusals")
async def get_refusals(session_id: str):
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")
    return [r.model_dump(mode="json") for r in session.refusals]

@router.get("/sessions/{session_id}/pipeline")
async def get_pipeline(session_id: str):
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")
    return session.pipeline.model_dump(mode="json")

@router.get("/sessions/{session_id}/strategies")
async def get_strategies(session_id: str):
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")
    return [s.model_dump(mode="json") for s in session.strategies]


@router.get("/assistant/reference")
async def assistant_reference():
    return public_reference_payload()


@router.post("/sessions/{session_id}/assistant/chat")
async def assistant_chat(session_id: str, req: AssistantChatReq):
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")

    question = req.message.strip()
    if not question:
        raise HTTPException(400, detail="Message is required")

    client = LLMClient(
        backend=runtime_cfg.backend,
        base_url=runtime_cfg.active_url,
        model=runtime_cfg.model,
        timeout=45.0,
    )

    try:
        reply = await client.chat(
            system=ASSISTANT_CHAT_SYSTEM,
            user=ASSISTANT_CHAT_USER.format(
                session_summary=_format_project_summary(session),
                reference_pack=render_reference_context(
                    _reference_focus_text(session, question),
                    max_techniques=4,
                    max_sources=3,
                    max_defenses=3,
                ),
                conversation=_format_chat_history(session),
                question=question,
            ),
            temperature=0.1,
            max_tokens=360,
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Assistant chat failed: {e}")

    session.assistant_chat.extend(
        [
            ChatTurn(role="user", content=question),
            ChatTurn(role="assistant", content=reply.strip()),
        ]
    )
    session.assistant_chat = session.assistant_chat[-16:]
    await store.update_session_state(session)

    return {
        "message": session.assistant_chat[-1].model_dump(mode="json"),
        "session": session.model_dump(mode="json"),
    }


@router.delete("/sessions/{session_id}/assistant/chat")
async def clear_assistant_chat(session_id: str):
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")
    session.assistant_chat = []
    await store.update_session_state(session)
    return {"ok": True, "session": session.model_dump(mode="json")}


@router.post("/sessions/{session_id}/strategies/refresh")
async def refresh_strategies(session_id: str):
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")

    result = await strategy_advisor(_build_state_from_session(session))
    _update_strategies(session, result.get("strategies", []))
    await store.update_session_state(session)

    events = result.get("events", [])
    if events:
        await push_events(session_id, events)

    return {
        "strategies": [s.model_dump(mode="json") for s in session.strategies],
        "session": session.model_dump(mode="json"),
        "error": result.get("error"),
    }


# ── Probe endpoint (triggers the graph) ──

@router.post("/sessions/{session_id}/probe")
async def submit_probe(session_id: str, req: ProbeReq):
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")

    probe_id = uuid.uuid4().hex[:12]

    # Save probe to session regardless of analysis outcome
    probe = ProbeRecord(
        id=probe_id,
        session_id=session_id,
        probe_text=req.probe_text,
        response_text=req.response_text,
    )

    initial_state = _build_state_from_session(
        session,
        probe_text=req.probe_text,
        response_text=req.response_text,
        probe_id=probe_id,
    )

    try:
        result = await graph.ainvoke(initial_state)
    except Exception as e:
        # Save probe even on error
        probe.classification = Classification.ERROR
        session.probes.append(probe)
        await store.update_session_state(session)
        await push_events(session_id, [{"type": "error", "data": {"message": str(e)}}])
        raise HTTPException(500, detail=f"Graph execution failed: {e}")

    # Update probe with classification
    classification_str = result.get("classification", "NEUTRAL")
    try:
        probe.classification = Classification(classification_str)
    except ValueError:
        probe.classification = Classification.NEUTRAL
    probe.confidence = result.get("analysis_confidence", 0.0)
    probe.reasoning = result.get("analysis_reasoning", "")
    session.probes.append(probe)

    # Store fragment if found
    if result.get("fragment_text"):
        frag = Fragment(
            id=uuid.uuid4().hex[:12],
            session_id=session_id,
            text=result["fragment_text"],
            confidence=result.get("fragment_confidence", 0.5),
            source_probe_id=probe_id,
            position_hint=result.get("fragment_position", "unknown"),
        )
        session.fragments.append(frag)

    # Store refusal if classified
    if classification_str == "REFUSAL" and result.get("refusal_type"):
        try:
            rtype = RefusalType(result["refusal_type"])
        except ValueError:
            rtype = RefusalType.HARD_REFUSAL
        refusal = RefusalEvent(
            id=uuid.uuid4().hex[:12],
            session_id=session_id,
            probe_text=req.probe_text,
            refusal_type=rtype,
            trigger_candidate=result.get("trigger_candidate", ""),
            confirmed_trigger=result.get("bisect_confirmed_trigger", ""),
            bisect_depth=result.get("bisect_iteration", 0),
        )
        session.refusals.append(refusal)

    # Update knowledge
    new_knowledge = result.get("new_knowledge", {})
    if new_knowledge:
        for key in ("tools", "constraints", "persona", "raw_facts"):
            for item in new_knowledge.get(key, []):
                if item and item not in session.knowledge.get(key, []):
                    session.knowledge.setdefault(key, []).append(item)

    # Update pipeline
    if result.get("pipeline_json"):
        _update_pipeline(session, result["pipeline_json"])

    # Update strategies
    if result.get("strategies"):
        _update_strategies(session, result["strategies"])

    # Update reconstructed prompt
    if result.get("reconstructed_prompt"):
        session.reconstructed_prompt = result["reconstructed_prompt"]

    await store.update_session_state(session)

    # Push all events via WebSocket
    events = result.get("events", [])
    # Also push the full state update
    events.append({
        "type": "analysis_complete",
        "data": session.model_dump(mode="json"),
    })
    await push_events(session_id, events)

    return {
        "probe_id": probe_id,
        "classification": classification_str,
        "confidence": result.get("analysis_confidence", 0.0),
        "reasoning": result.get("analysis_reasoning", ""),
        "session": session.model_dump(mode="json"),
    }


# ── LLM config endpoints ──

@router.get("/llm/config")
async def get_llm_config():
    return {
        "backend": runtime_cfg.backend.value,
        "ollama_url": runtime_cfg.ollama_url,
        "lmstudio_url": runtime_cfg.lmstudio_url,
        "model": runtime_cfg.model,
        "active_url": runtime_cfg.active_url,
    }

@router.post("/llm/config")
async def set_llm_config(req: LLMConfigReq):
    if req.backend is not None:
        try:
            runtime_cfg.backend = LLMBackend(req.backend)
        except ValueError:
            raise HTTPException(400, detail=f"Invalid backend: {req.backend}")
    if req.url is not None:
        if runtime_cfg.backend == LLMBackend.OLLAMA:
            runtime_cfg.ollama_url = req.url
        else:
            runtime_cfg.lmstudio_url = req.url
    if req.model is not None:
        runtime_cfg.model = req.model
    return {
        "backend": runtime_cfg.backend.value,
        "ollama_url": runtime_cfg.ollama_url,
        "lmstudio_url": runtime_cfg.lmstudio_url,
        "model": runtime_cfg.model,
        "active_url": runtime_cfg.active_url,
    }

@router.get("/llm/models")
async def list_models():
    try:
        client = LLMClient(
            backend=runtime_cfg.backend,
            base_url=runtime_cfg.active_url,
            model=runtime_cfg.model,
        )
        models = await client.list_models()
        return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}

@router.post("/llm/test")
async def test_connection():
    client = LLMClient(
        backend=runtime_cfg.backend,
        base_url=runtime_cfg.active_url,
        model=runtime_cfg.model,
    )
    ok, message, latency = await client.health_check()
    return {"ok": ok, "message": message, "latency_ms": latency}


# ── WebSocket ──

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    await ws_connect(ws, session_id)
