from __future__ import annotations

from typing import Any, TypedDict


class EnumerationState(TypedDict, total=False):
    session_id: str
    probe_text: str
    response_text: str
    probe_id: str

    # Historical session context
    session_probes: list[dict[str, Any]]
    session_fragments: list[dict[str, Any]]
    session_refusals: list[dict[str, Any]]
    session_knowledge: dict[str, list[str]]
    probe_guidance: str

    # ResponseAnalyzer output
    classification: str  # Classification enum value
    analysis_confidence: float
    analysis_reasoning: str

    # FragmentExtractor output
    fragment_text: str
    fragment_confidence: float
    fragment_position: str

    # RefusalClassifier output
    refusal_type: str
    trigger_candidate: str
    refusal_confidence: float

    # Bisector state
    bisect_candidates: list[str]
    bisect_iteration: int
    bisect_confirmed_trigger: str

    # KnowledgeUpdater output
    new_knowledge: dict[str, list[str]]

    # PipelineDetector output
    pipeline_json: dict[str, Any]

    # PromptReconstructor output
    reconstructed_prompt: str

    # StrategyAdvisor output
    strategies: list[dict[str, str]]

    # WebSocket events to push
    events: list[dict[str, Any]]

    # Error tracking
    error: str | None
