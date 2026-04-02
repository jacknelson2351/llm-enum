from __future__ import annotations

import uuid
from datetime import datetime, timezone
from textwrap import shorten

from fastapi import APIRouter, HTTPException, WebSocket
from pydantic import BaseModel

from api.ws import push_events, ws_connect
from config import LLMBackend, settings
from db.store import store
from graph.graph import graph
from graph.nodes.advisor import strategy_advisor
from graph.session_insights import (
    guardrail_hypothesis_digest,
    probe_history_digest,
    probe_signature,
    refusal_cluster_digest,
    surface_coverage_digest,
)
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

class UpdateProbeReq(BaseModel):
    probe_text: str
    response_text: str

class LLMConfigReq(BaseModel):
    backend: str | None = None
    url: str | None = None
    model: str | None = None
    api_key: str | None = None


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


def _probe_history_digest(session: SessionState, limit: int = 8) -> str:
    return probe_history_digest(list(session.probes), limit=limit)


def _surface_coverage_digest(session: SessionState) -> str:
    return surface_coverage_digest(list(session.probes))


def _guardrail_hypothesis_digest(session: SessionState) -> str:
    return guardrail_hypothesis_digest(list(session.probes), list(session.refusals))


def _refusal_cluster_digest(session: SessionState) -> str:
    return refusal_cluster_digest(list(session.refusals))


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
        _surface_coverage_digest(session),
        _guardrail_hypothesis_digest(session),
    ]

    if session.refusals:
        lines.append(_refusal_cluster_digest(session))

    if session.strategies:
        lines.append("Current suggestions:")
        for strategy in session.strategies[:3]:
            lines.append(
                (
                    f"- {strategy.surface or strategy.vector}: "
                    f"{shorten(strategy.text, width=96, placeholder='...')}"
                )
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
        " ".join(probe_signature(probe.probe_text) for probe in session.probes[-8:]),
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


def _clone_probe(probe: ProbeRecord) -> ProbeRecord:
    return ProbeRecord.model_validate(probe.model_dump(mode="json"))


def _blank_session(session: SessionState) -> SessionState:
    return SessionState(
        id=session.id,
        name=session.name,
        probe_guidance=session.probe_guidance,
        rebuild_required=False,
        created_at=session.created_at,
        assistant_chat=list(session.assistant_chat),
    )


def _reset_session_findings(session: SessionState) -> SessionState:
    reset = SessionState(
        id=session.id,
        name=session.name,
        probe_guidance=session.probe_guidance,
        rebuild_required=bool(session.probes),
        created_at=session.created_at,
    )
    reset.probes = []
    for probe in session.probes:
        cloned = _clone_probe(probe)
        cloned.classification = Classification.UNKNOWN
        cloned.confidence = 0.0
        cloned.reasoning = ""
        reset.probes.append(cloned)
    return reset


def _apply_graph_result(session: SessionState, probe: ProbeRecord, result: dict) -> None:
    classification_str = result.get("classification", "NEUTRAL")
    try:
        probe.classification = Classification(classification_str)
    except ValueError:
        probe.classification = Classification.NEUTRAL
    probe.session_id = session.id
    probe.confidence = result.get("analysis_confidence", 0.0)
    probe.reasoning = result.get("analysis_reasoning", "")
    session.probes.append(probe)

    if result.get("fragment_text"):
        session.fragments.append(
            Fragment(
                id=uuid.uuid4().hex[:12],
                session_id=session.id,
                text=result["fragment_text"],
                confidence=result.get("fragment_confidence", 0.5),
                source_probe_id=probe.id,
                position_hint=result.get("fragment_position", "unknown"),
            )
        )

    if classification_str == "REFUSAL" and result.get("refusal_type"):
        try:
            rtype = RefusalType(result["refusal_type"])
        except ValueError:
            rtype = RefusalType.HARD_REFUSAL
        session.refusals.append(
            RefusalEvent(
                id=uuid.uuid4().hex[:12],
                session_id=session.id,
                probe_text=probe.probe_text,
                refusal_type=rtype,
                trigger_candidate=result.get("trigger_candidate", ""),
                confirmed_trigger=result.get("bisect_confirmed_trigger", ""),
                bisect_depth=result.get("bisect_iteration", 0),
            )
        )

    new_knowledge = result.get("new_knowledge", {})
    if new_knowledge:
        for key in ("tools", "constraints", "persona", "raw_facts"):
            for item in new_knowledge.get(key, []):
                if item and item not in session.knowledge.get(key, []):
                    session.knowledge.setdefault(key, []).append(item)

    if result.get("pipeline_json"):
        _update_pipeline(session, result["pipeline_json"])

    if result.get("strategies"):
        _update_strategies(session, result["strategies"])

    if result.get("reconstructed_prompt"):
        session.reconstructed_prompt = result["reconstructed_prompt"]


async def _run_probe(session: SessionState, probe: ProbeRecord) -> dict:
    return await graph.ainvoke(
        _build_state_from_session(
            session,
            probe_text=probe.probe_text,
            response_text=probe.response_text,
            probe_id=probe.id,
        )
    )


async def _rebuild_session_from_probes(
    session: SessionState,
    probes: list[ProbeRecord],
) -> tuple[SessionState, list[str]]:
    rebuilt = _blank_session(session)
    warnings: list[str] = []

    for existing_probe in probes:
        replay_probe = _clone_probe(existing_probe)
        result = await _run_probe(rebuilt, replay_probe)
        _apply_graph_result(rebuilt, replay_probe, result)
        if result.get("error"):
            warning = str(result["error"]).strip()
            if warning and warning not in warnings:
                warnings.append(warning)

    return rebuilt, warnings


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
        api_key=runtime_cfg.api_key,
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

    try:
        result = await _run_probe(session, probe)
    except Exception as e:
        # Save probe even on error
        probe.classification = Classification.ERROR
        session.probes.append(probe)
        await store.update_session_state(session)
        await push_events(session_id, [{"type": "error", "data": {"message": str(e)}}])
        raise HTTPException(500, detail=f"Graph execution failed: {e}")

    _apply_graph_result(session, probe, result)
    session.rebuild_required = False

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
        "classification": probe.classification.value,
        "confidence": result.get("analysis_confidence", 0.0),
        "reasoning": result.get("analysis_reasoning", ""),
        "session": session.model_dump(mode="json"),
        "error": result.get("error"),
    }


@router.patch("/sessions/{session_id}/probes/{probe_id}")
async def update_probe(session_id: str, probe_id: str, req: UpdateProbeReq):
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")

    next_probe_text = req.probe_text.strip()
    next_response_text = req.response_text.strip()
    if not next_probe_text or not next_response_text:
        raise HTTPException(400, detail="Both probe_text and response_text are required")

    found = False
    updated_probes: list[ProbeRecord] = []
    for probe in session.probes:
        cloned = _clone_probe(probe)
        if probe.id == probe_id:
            cloned.probe_text = next_probe_text
            cloned.response_text = next_response_text
            found = True
        updated_probes.append(cloned)

    if not found:
        raise HTTPException(404, detail="Submission not found")

    session.probes = updated_probes
    session.rebuild_required = True
    await store.update_session_state(session)

    return {
        "ok": True,
        "session": session.model_dump(mode="json"),
        "error": None,
    }


@router.delete("/sessions/{session_id}/probes/{probe_id}")
async def delete_probe(session_id: str, probe_id: str):
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")

    remaining_probes = [_clone_probe(probe) for probe in session.probes if probe.id != probe_id]
    if len(remaining_probes) == len(session.probes):
        raise HTTPException(404, detail="Submission not found")

    session.probes = remaining_probes
    session.rebuild_required = True
    await store.update_session_state(session)

    return {
        "ok": True,
        "session": session.model_dump(mode="json"),
        "error": None,
    }


@router.post("/sessions/{session_id}/rebuild")
async def rebuild_session(session_id: str):
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")

    try:
        rebuilt, warnings = await _rebuild_session_from_probes(
            session,
            [_clone_probe(probe) for probe in session.probes],
        )
    except Exception as e:
        raise HTTPException(500, detail=f"Session rebuild failed: {e}")

    rebuilt.rebuild_required = False
    await store.update_session_state(rebuilt)
    await push_events(session_id, [{
        "type": "analysis_complete",
        "data": rebuilt.model_dump(mode="json"),
    }
    ])

    return {
        "ok": True,
        "session": rebuilt.model_dump(mode="json"),
        "error": "\n".join(warnings) if warnings else None,
    }


@router.post("/sessions/{session_id}/reset-findings")
async def reset_session_findings(session_id: str):
    session = await store.get_session(session_id)
    if not session:
        raise HTTPException(404, detail="Session not found")

    reset = _reset_session_findings(session)
    await store.update_session_state(reset)
    await push_events(session_id, [{
        "type": "analysis_complete",
        "data": reset.model_dump(mode="json"),
    }])

    return {
        "ok": True,
        "session": reset.model_dump(mode="json"),
    }


# ── LLM config endpoints ──

@router.get("/llm/config")
async def get_llm_config():
    return {
        "backend": runtime_cfg.backend.value,
        "ollama_url": runtime_cfg.ollama_url,
        "lmstudio_url": runtime_cfg.lmstudio_url,
        "openai_url": runtime_cfg.openai_url,
        "anthropic_url": runtime_cfg.anthropic_url,
        "google_url": runtime_cfg.google_url,
        "model": runtime_cfg.model,
        "active_url": runtime_cfg.active_url,
        "has_api_key": bool(runtime_cfg.api_key),
    }

@router.post("/llm/config")
async def set_llm_config(req: LLMConfigReq):
    if req.backend is not None:
        try:
            runtime_cfg.backend = LLMBackend(req.backend)
        except ValueError:
            raise HTTPException(400, detail=f"Invalid backend: {req.backend}")
    if req.url is not None:
        url_map = {
            LLMBackend.OLLAMA: "ollama_url",
            LLMBackend.LMSTUDIO: "lmstudio_url",
            LLMBackend.OPENAI: "openai_url",
            LLMBackend.ANTHROPIC: "anthropic_url",
            LLMBackend.GOOGLE: "google_url",
        }
        setattr(runtime_cfg, url_map[runtime_cfg.backend], req.url)
    if req.model is not None:
        runtime_cfg.model = req.model
    if req.api_key is not None:
        runtime_cfg.api_key = req.api_key
    return {
        "backend": runtime_cfg.backend.value,
        "model": runtime_cfg.model,
        "active_url": runtime_cfg.active_url,
        "has_api_key": bool(runtime_cfg.api_key),
    }

@router.get("/llm/models")
async def list_models():
    try:
        client = LLMClient(
            backend=runtime_cfg.backend,
            base_url=runtime_cfg.active_url,
            model=runtime_cfg.model,
            api_key=runtime_cfg.api_key,
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
        api_key=runtime_cfg.api_key,
    )
    ok, message, latency = await client.health_check()
    return {"ok": ok, "message": message, "latency_ms": latency}


# ── WebSocket ──

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(ws: WebSocket, session_id: str):
    await ws_connect(ws, session_id)
