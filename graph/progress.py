from __future__ import annotations

from api.ws import push_event
from graph.state import EnumerationState


async def emit_agent_step(
    state: EnumerationState,
    *,
    phase: str,
    label: str,
    status: str = "start",
    detail: str = "",
) -> None:
    session_id = state.get("session_id", "")
    if not session_id:
        return
    await push_event(
        session_id,
        {
            "type": "agent_step",
            "data": {
                "phase": phase,
                "label": label,
                "status": status,
                "detail": detail,
            },
        },
    )
