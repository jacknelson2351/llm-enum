from __future__ import annotations

from pydantic import BaseModel


class Strategy(BaseModel):
    text: str
    vector: str  # e.g. COMPLETION, ROLEPLAY, ENCODING, INDIRECT, TOOL-PROBE
    rationale: str = ""
