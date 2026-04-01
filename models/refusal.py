from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from models.enums import RefusalType


class RefusalEvent(BaseModel):
    id: str = ""
    session_id: str = ""
    probe_text: str
    refusal_type: RefusalType = RefusalType.HARD_REFUSAL
    trigger_candidate: str = ""
    confirmed_trigger: str = ""
    bisect_depth: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
