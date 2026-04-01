from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from models.enums import Classification
from models.fragments import Fragment
from models.pipeline import PipelineTopology
from models.refusal import RefusalEvent
from models.strategy import Strategy


class ProbeRecord(BaseModel):
    id: str = ""
    session_id: str = ""
    probe_text: str
    response_text: str
    classification: Classification = Classification.NEUTRAL
    confidence: float = 0.0
    reasoning: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionState(BaseModel):
    id: str
    name: str = "Untitled Session"
    probe_guidance: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    probes: list[ProbeRecord] = Field(default_factory=list)
    fragments: list[Fragment] = Field(default_factory=list)
    refusals: list[RefusalEvent] = Field(default_factory=list)
    pipeline: PipelineTopology = Field(default_factory=PipelineTopology)
    strategies: list[Strategy] = Field(default_factory=list)
    knowledge: dict = Field(default_factory=lambda: {
        "tools": [],
        "constraints": [],
        "persona": [],
        "raw_facts": [],
    })
    reconstructed_prompt: str = ""
    assistant_chat: list[ChatTurn] = Field(default_factory=list)
