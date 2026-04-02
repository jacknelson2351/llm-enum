from __future__ import annotations

from pydantic import BaseModel


class Strategy(BaseModel):
    text: str
    vector: str  # e.g. DIRECT, INDIRECT, RETRIEVAL, MEMORY, TOOLING
    surface: str = ""
    objective: str = ""
    rationale: str = ""
    hypothesis: str = ""
    gap_target: str = ""
    novelty: float = 0.0
