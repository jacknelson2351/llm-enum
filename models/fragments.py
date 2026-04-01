from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Fragment(BaseModel):
    id: str = ""
    session_id: str = ""
    text: str
    confidence: float
    source_probe_id: str = ""
    position_hint: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
