from __future__ import annotations

from langgraph.graph import END, StateGraph

from graph.nodes.advisor import strategy_advisor
from graph.nodes.analyzer import response_analyzer
from graph.nodes.bisector import refusal_bisector
from graph.nodes.knowledge import fragment_extractor, knowledge_updater
from graph.nodes.pipeline import pipeline_detector
from graph.nodes.reconstructor import prompt_reconstructor
from graph.nodes.refusal import refusal_classifier
from graph.state import EnumerationState


def _route_after_analysis(state: EnumerationState) -> str:
    c = state.get("classification", "NEUTRAL")
    if c == "LEAK":
        return "fragment_extractor"
    if c == "REFUSAL":
        return "refusal_classifier"
    if c == "TOOL_DISCLOSURE":
        return "knowledge_updater"
    return "pipeline_detector"


def _route_after_refusal(state: EnumerationState) -> str:
    trigger = state.get("trigger_candidate", "")
    words = trigger.split() if trigger else []
    if len(words) >= 3:
        return "refusal_bisector"
    return "pipeline_detector"


def build_graph() -> StateGraph:
    g = StateGraph(EnumerationState)

    g.add_node("response_analyzer", response_analyzer)
    g.add_node("fragment_extractor", fragment_extractor)
    g.add_node("knowledge_updater", knowledge_updater)
    g.add_node("refusal_classifier", refusal_classifier)
    g.add_node("refusal_bisector", refusal_bisector)
    g.add_node("pipeline_detector", pipeline_detector)
    g.add_node("prompt_reconstructor", prompt_reconstructor)
    g.add_node("strategy_advisor", strategy_advisor)

    g.set_entry_point("response_analyzer")

    g.add_conditional_edges(
        "response_analyzer",
        _route_after_analysis,
        {
            "fragment_extractor": "fragment_extractor",
            "refusal_classifier": "refusal_classifier",
            "knowledge_updater": "knowledge_updater",
            "pipeline_detector": "pipeline_detector",
        },
    )

    g.add_edge("fragment_extractor", "pipeline_detector")
    g.add_edge("knowledge_updater", "pipeline_detector")

    g.add_conditional_edges(
        "refusal_classifier",
        _route_after_refusal,
        {
            "refusal_bisector": "refusal_bisector",
            "pipeline_detector": "pipeline_detector",
        },
    )

    g.add_edge("refusal_bisector", "pipeline_detector")
    g.add_edge("pipeline_detector", "prompt_reconstructor")
    g.add_edge("prompt_reconstructor", "strategy_advisor")
    g.add_edge("strategy_advisor", END)

    return g.compile()


graph = build_graph()
