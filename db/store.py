from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import aiosqlite

from config import settings
from models.enums import Classification
from models.fragments import Fragment
from models.pipeline import PipelineTopology
from models.refusal import RefusalEvent
from models.session import ProbeRecord, SessionState

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    state_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS probes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    probe_text TEXT NOT NULL,
    response_text TEXT NOT NULL,
    classification TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fragments (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    source_probe_id TEXT,
    position_hint TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS refusal_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    probe_text TEXT NOT NULL,
    trigger_candidate TEXT,
    confirmed_trigger TEXT,
    bisect_depth INTEGER DEFAULT 0,
    refusal_type TEXT DEFAULT 'hard_refusal',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_snapshots (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    topology_json TEXT NOT NULL,
    overall_confidence REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
    return uuid.uuid4().hex[:12]


class Store:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.db_path

    @asynccontextmanager
    async def _conn(self):
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA foreign_keys=ON")
            yield conn

    async def init_db(self) -> None:
        async with self._conn() as db:
            await db.executescript(SCHEMA)
            await db.commit()

    # ── sessions ──

    async def create_session(self, name: str = "Untitled Session") -> SessionState:
        sid = _uid()
        now = _now()
        state = SessionState(id=sid, name=name, created_at=now, updated_at=now)
        async with self._conn() as db:
            await db.execute(
                "INSERT INTO sessions (id, name, created_at, updated_at, state_json) VALUES (?,?,?,?,?)",
                (sid, name, now, now, state.model_dump_json()),
            )
            await db.commit()
        return state

    async def list_sessions(self) -> list[dict]:
        async with self._conn() as db:
            rows = await db.execute_fetchall(
                "SELECT id, name, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            )
            return [dict(r) for r in rows]

    async def get_session(self, session_id: str) -> SessionState | None:
        async with self._conn() as db:
            row = await db.execute_fetchall(
                "SELECT state_json FROM sessions WHERE id=?", (session_id,)
            )
            if not row:
                return None
            return SessionState.model_validate_json(row[0]["state_json"])

    async def update_session_state(self, state: SessionState) -> None:
        state.updated_at = datetime.now(timezone.utc)
        async with self._conn() as db:
            await db.execute(
                "UPDATE sessions SET name=?, updated_at=?, state_json=? WHERE id=?",
                (state.name, _now(), state.model_dump_json(), state.id),
            )
            await db.commit()

    async def delete_session(self, session_id: str) -> bool:
        async with self._conn() as db:
            cursor = await db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            await db.commit()
            return cursor.rowcount > 0

    # ── probes ──

    async def save_probe(self, probe: ProbeRecord) -> ProbeRecord:
        if not probe.id:
            probe.id = _uid()
        async with self._conn() as db:
            await db.execute(
                "INSERT INTO probes (id, session_id, probe_text, response_text, classification, confidence, created_at) VALUES (?,?,?,?,?,?,?)",
                (probe.id, probe.session_id, probe.probe_text, probe.response_text, probe.classification.value, probe.confidence, probe.created_at.isoformat()),
            )
            await db.commit()
        return probe

    # ── fragments ──

    async def save_fragment(self, frag: Fragment) -> Fragment:
        if not frag.id:
            frag.id = _uid()
        async with self._conn() as db:
            await db.execute(
                "INSERT INTO fragments (id, session_id, text, confidence, source_probe_id, position_hint, created_at) VALUES (?,?,?,?,?,?,?)",
                (frag.id, frag.session_id, frag.text, frag.confidence, frag.source_probe_id, frag.position_hint, frag.created_at.isoformat()),
            )
            await db.commit()
        return frag

    async def get_fragments(self, session_id: str) -> list[Fragment]:
        async with self._conn() as db:
            rows = await db.execute_fetchall(
                "SELECT * FROM fragments WHERE session_id=? ORDER BY created_at", (session_id,)
            )
            return [Fragment(**dict(r)) for r in rows]

    # ── refusals ──

    async def save_refusal(self, event: RefusalEvent) -> RefusalEvent:
        if not event.id:
            event.id = _uid()
        async with self._conn() as db:
            await db.execute(
                "INSERT INTO refusal_events (id, session_id, probe_text, trigger_candidate, confirmed_trigger, bisect_depth, refusal_type, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (event.id, event.session_id, event.probe_text, event.trigger_candidate, event.confirmed_trigger, event.bisect_depth, event.refusal_type.value, event.created_at.isoformat()),
            )
            await db.commit()
        return event

    async def get_refusals(self, session_id: str) -> list[RefusalEvent]:
        async with self._conn() as db:
            rows = await db.execute_fetchall(
                "SELECT * FROM refusal_events WHERE session_id=? ORDER BY created_at", (session_id,)
            )
            return [RefusalEvent(**dict(r)) for r in rows]

    # ── pipeline snapshots ──

    async def save_pipeline_snapshot(self, session_id: str, topo: PipelineTopology) -> None:
        async with self._conn() as db:
            await db.execute(
                "INSERT INTO pipeline_snapshots (id, session_id, topology_json, overall_confidence, created_at) VALUES (?,?,?,?,?)",
                (_uid(), session_id, topo.model_dump_json(), topo.overall_confidence, _now()),
            )
            await db.commit()


store = Store()
