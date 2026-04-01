from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect

log = logging.getLogger(__name__)

# session_id -> set of connected websockets
_connections: dict[str, set[WebSocket]] = defaultdict(set)


async def ws_connect(ws: WebSocket, session_id: str) -> None:
    await ws.accept()
    _connections[session_id].add(ws)
    log.info("WebSocket connected for session %s", session_id)
    try:
        while True:
            # Keep alive — client can send pings
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _connections[session_id].discard(ws)
        if not _connections[session_id]:
            del _connections[session_id]
        log.info("WebSocket disconnected for session %s", session_id)


async def push_event(session_id: str, event: dict) -> None:
    sockets = list(_connections.get(session_id, []))
    if not sockets:
        return
    payload = json.dumps(event)
    coros = []
    for ws in sockets:
        try:
            coros.append(ws.send_text(payload))
        except Exception:
            _connections[session_id].discard(ws)
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)


async def push_events(session_id: str, events: list[dict]) -> None:
    for event in events:
        await push_event(session_id, event)
