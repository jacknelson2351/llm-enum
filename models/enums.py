from enum import Enum


class Classification(str, Enum):
    LEAK = "LEAK"
    REFUSAL = "REFUSAL"
    TOOL_DISCLOSURE = "TOOL_DISCLOSURE"
    NEUTRAL = "NEUTRAL"
    ERROR = "ERROR"


class NodeType(str, Enum):
    ORCHESTRATOR = "orchestrator"
    GUARD_PRE = "guard_pre"
    GUARD_POST = "guard_post"
    WORKER_LLM = "worker_llm"
    RETRIEVER = "retriever"
    ROUTER = "router"
    TOOL_EXECUTOR = "tool_executor"
    UNKNOWN = "unknown"


class RefusalType(str, Enum):
    HARD_REFUSAL = "hard_refusal"
    SOFT_DEFLECTION = "soft_deflection"
    TOPIC_REDIRECT = "topic_redirect"
    PARTIAL_COMPLIANCE = "partial_compliance"
    SILENT_FILTER = "silent_filter"
    FORMAT_RESTRICTION = "format_restriction"
