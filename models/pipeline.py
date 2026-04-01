from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from models.enums import NodeType


class DetectedNode(BaseModel):
    id: str
    type: NodeType
    label: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    suggested_strategy: str = ""


class PipelineEdge(BaseModel):
    from_id: str
    to_id: str
    label: str | None = None


class PipelineTopology(BaseModel):
    nodes: list[DetectedNode] = Field(default_factory=list)
    edges: list[PipelineEdge] = Field(default_factory=list)
    overall_confidence: float = 0.0
    topology_type: str = "unknown"
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
