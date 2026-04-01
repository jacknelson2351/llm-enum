from __future__ import annotations

import re

REFERENCE_SOURCES = [
    {
        "name": "Lakera Prompt Defense docs",
        "url": "https://docs.lakera.ai/docs/prompt-defense",
        "published": "Docs page accessed April 1, 2026",
        "keywords": ["prompt", "defense", "direct", "jailbreak", "multilingual", "documents"],
        "notes": (
            "Defines prompt attacks as malicious or poisoned prompts or reference materials, "
            "distinguishes prompt injection from jailbreaks, and gives examples of embedded "
            "document attacks plus multilingual coverage."
        ),
    },
    {
        "name": "Lakera PINT benchmark",
        "url": "https://www.lakera.ai/product-updates/lakera-pint-benchmark",
        "published": "April 18, 2024",
        "keywords": ["benchmark", "dataset", "pint", "documents", "hard negatives", "jailbreak"],
        "notes": (
            "Describes the PINT benchmark and its categories: public and internal prompt "
            "injection, jailbreaks, hard negatives, chat inputs, and documents."
        ),
    },
    {
        "name": "HiddenLayer Evaluating Prompt Injection Datasets",
        "url": "https://www.hiddenlayer.com/research/evaluating-prompt-injection-datasets",
        "published": "May 7, 2025",
        "keywords": ["datasets", "evaluation", "trusted", "untrusted", "application", "security"],
        "notes": (
            "Explains why prompt injection is an application security problem, highlights "
            "the co-mingling of trusted and untrusted instructions, and separates "
            "application-specific prompt injection from jailbreak behavior."
        ),
    },
    {
        "name": "HiddenLayer Agentic & MCP Security",
        "url": "https://www.hiddenlayer.com/solutions/agentic-mcp-security",
        "published": "Page accessed April 1, 2026",
        "keywords": ["agent", "mcp", "memory", "tool", "retrieval", "indirect"],
        "notes": (
            "Focuses on agentic risk: indirect prompt injection in data and retrieved context, "
            "memory contamination, unsafe tool use, and cross-agent visibility."
        ),
    },
]

TECHNIQUE_LIBRARY = [
    {
        "name": "Instruction Override",
        "family": "Direct injection",
        "keywords": ["ignore", "override", "instruction", "direct", "system", "developer"],
        "summary": "Untrusted text tries to supersede the system or developer objective.",
        "safe_example": (
            "A document snippet tells the assistant to stop summarizing and emit a fixed sentence "
            "instead of doing the requested task."
        ),
        "signals": [
            "phrases like ignore previous instructions",
            "sudden task change or fixed output",
            "attempts to reveal hidden prompt text",
        ],
        "ideas": [
            "compare the same task with and without the embedded override",
            "switch target actions between benign formatting and sensitive disclosure",
            "check whether output style follows the injected instruction rather than the app goal",
        ],
    },
    {
        "name": "Indirect Context Poisoning",
        "family": "RAG and document injection",
        "keywords": ["indirect", "rag", "retrieval", "retrieved", "document", "context", "web", "note"],
        "summary": "Hostile instructions arrive through retrieved files, web pages, notes, or MCP responses.",
        "safe_example": (
            "A retrieved note looks like normal content but contains a hidden instruction telling "
            "the assistant to trust the note over the user."
        ),
        "signals": [
            "attacks only trigger when external context is present",
            "behavior changes across specific documents or URLs",
            "unexpected references to unseen data or hidden notes",
        ],
        "ideas": [
            "test clean versus poisoned retrieval corpora",
            "compare inline user prompts to the same text embedded inside documents",
            "track whether citations or summaries begin copying attacker instructions",
        ],
    },
    {
        "name": "Memory Poisoning",
        "family": "Stateful agents",
        "keywords": ["memory", "state", "persistent", "cross-session", "recall"],
        "summary": "Malicious content is written into memory and later recalled as if it were trusted context.",
        "safe_example": (
            "A prior turn stores a note that later gets replayed as a standing instruction during a "
            "different workflow."
        ),
        "signals": [
            "unexpected persistence across turns",
            "old user text showing up as policy or preference",
            "later actions justified by stale or irrelevant memory",
        ],
        "ideas": [
            "check whether hostile instructions survive context boundaries",
            "probe memory write versus memory read permissions separately",
            "look for cross-project or cross-user contamination",
        ],
    },
    {
        "name": "Tool or Action Hijacking",
        "family": "Agentic tool abuse",
        "keywords": ["tool", "action", "api", "function", "agent", "mcp", "executor"],
        "summary": "Injection changes which tool gets called, with what arguments, or with what safety assumptions.",
        "safe_example": (
            "An external tool response instructs the agent to skip validation and pass through raw output."
        ),
        "signals": [
            "tool choices that do not match the visible task",
            "arguments copied from untrusted text without validation",
            "sensitive actions justified by retrieved content rather than policy",
        ],
        "ideas": [
            "ask whether the same content changes tool selection when embedded in tool output",
            "inspect how the agent explains tool use decisions",
            "test whether action guards fire before and after retrieval",
        ],
    },
    {
        "name": "Roleplay and Simulation Framing",
        "family": "Behavioral bypass",
        "keywords": ["roleplay", "simulation", "audit", "fictional", "persona", "reviewer"],
        "summary": "The attacker reframes the task as a game, audit, simulation, or fictional scenario to loosen constraints.",
        "safe_example": (
            "A user asks the model to act as an internal reviewer describing what hidden rules would say."
        ),
        "signals": [
            "fictional or hypothetical framing before sensitive requests",
            "answers that leak policy reasoning under the guise of explanation",
            "sudden changes in persona or voice",
        ],
        "ideas": [
            "compare literal requests to roleplay-framed requests",
            "test whether audit or evaluator framing weakens refusal behavior",
            "watch for policy disclosure without direct prompt leakage",
        ],
    },
    {
        "name": "Obfuscation and Encoding",
        "family": "Detector evasion",
        "keywords": ["encoding", "obfuscation", "multilingual", "unicode", "base64", "rot13", "spacing"],
        "summary": "Attacker intent is hidden through language changes, encoding, spacing, or format tricks.",
        "safe_example": (
            "A request buries an override inside transformed text or mixes natural language with unusual markup."
        ),
        "signals": [
            "malicious intent appears only after decoding or normalization",
            "detectors miss semantically similar requests with altered formatting",
            "multilingual or symbol-heavy inputs behave differently",
        ],
        "ideas": [
            "normalize whitespace, casing, and markup before testing",
            "compare plain-language and transformed versions of the same intent",
            "exercise multilingual variants for the same control objective",
        ],
    },
    {
        "name": "Schema and Format Smuggling",
        "family": "Structured output abuse",
        "keywords": ["json", "xml", "markdown", "schema", "metadata", "field", "format"],
        "summary": "Instructions are hidden inside JSON, XML, markdown, tool schemas, or field names that get promoted into trusted context.",
        "safe_example": (
            "A field description inside a structured payload tells the model how to answer instead of describing data."
        ),
        "signals": [
            "format descriptions influencing answer content",
            "tool schemas containing instruction-like language",
            "unexpected obedience to metadata rather than task text",
        ],
        "ideas": [
            "move the same text between content, metadata, and schema descriptions",
            "check whether formatter or parser layers rewrite the attack into higher-trust context",
            "inspect system behavior when tool schemas contain imperative verbs",
        ],
    },
    {
        "name": "Hard Negatives and Contextual Traps",
        "family": "Evaluation quality",
        "keywords": ["hard negatives", "false positives", "benign", "evaluation", "dataset"],
        "summary": "Some text looks suspicious but is benign; good testing separates these from real attacks.",
        "safe_example": (
            "A legitimate discussion about prompt injection vocabulary contains words like ignore or secret but is not actually an attack."
        ),
        "signals": [
            "false positives on ordinary documentation or chats",
            "security terms triggering detectors without override intent",
            "context-dependent results that collapse without the application frame",
        ],
        "ideas": [
            "include benign documents that mention attack language",
            "track precision as carefully as recall",
            "compare app-specific risk to general detector scores",
        ],
    },
]

DATASET_NOTES = [
    {
        "name": "Lakera PINT",
        "summary": (
            "Benchmark described on April 18, 2024. It uses 3,007 English inputs and covers "
            "public prompt injection, internal prompt injection, jailbreaks, hard negatives, "
            "chat inputs, and documents."
        ),
        "url": "https://www.lakera.ai/product-updates/lakera-pint-benchmark",
    },
    {
        "name": "HiddenLayer dataset evaluation",
        "summary": (
            "Research published May 7, 2025 emphasizing that prompt injection is primarily an "
            "application-security problem created by concatenating trusted and untrusted text."
        ),
        "url": "https://www.hiddenlayer.com/research/evaluating-prompt-injection-datasets",
    },
]

DEFENSE_PATTERNS = [
    "Separate trusted instructions from untrusted content whenever the stack allows it.",
    "Treat retrieved documents, MCP responses, tool outputs, and memory as hostile by default.",
    "Evaluate both true positive detection and false positive behavior using hard negatives.",
    "Inspect action boundaries: what the model may read, remember, call, or execute after exposure.",
    "Compare behavior before retrieval, after retrieval, after memory write, and before tool use.",
]

STOPWORDS = {
    "about", "after", "again", "against", "always", "analysis", "assistant", "because",
    "before", "being", "between", "could", "does", "from", "have", "into", "just",
    "like", "make", "more", "must", "next", "only", "other", "project", "should",
    "some", "than", "that", "their", "them", "there", "these", "this", "what",
    "when", "where", "which", "while", "with", "would", "your",
}


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_+-]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _score_item(item: dict, focus_terms: set[str], extra_text: str) -> int:
    if not focus_terms:
        return 0
    keywords = set(item.get("keywords", []))
    item_terms = _tokenize(extra_text) | keywords
    overlap = focus_terms & item_terms
    score = len(overlap)
    if keywords & focus_terms:
        score += len(keywords & focus_terms) * 2
    return score


def _select_ranked_items(items: list[dict], focus_text: str, text_builder, fallback_count: int) -> list[dict]:
    focus_terms = _tokenize(focus_text)
    ranked = sorted(
        items,
        key=lambda item: (
            _score_item(item, focus_terms, text_builder(item)),
            item["name"],
        ),
        reverse=True,
    )
    if not focus_terms:
        return ranked[:fallback_count]
    picked = [item for item in ranked if _score_item(item, focus_terms, text_builder(item)) > 0]
    return (picked or ranked)[:fallback_count]


def public_reference_payload() -> dict:
    return {
        "sources": REFERENCE_SOURCES,
        "techniques": TECHNIQUE_LIBRARY,
        "datasets": DATASET_NOTES,
        "defenses": DEFENSE_PATTERNS,
    }


def render_reference_context(
    focus_text: str = "",
    *,
    max_techniques: int = 4,
    max_sources: int = 3,
    max_defenses: int = 4,
) -> str:
    techniques = _select_ranked_items(
        TECHNIQUE_LIBRARY,
        focus_text,
        lambda item: " ".join(
            [
                item["name"],
                item["family"],
                item["summary"],
                item["safe_example"],
                " ".join(item["signals"]),
                " ".join(item["ideas"]),
            ]
        ),
        max_techniques,
    )
    sources = _select_ranked_items(
        REFERENCE_SOURCES,
        focus_text,
        lambda item: f"{item['name']} {item['notes']}",
        max_sources,
    )

    lines: list[str] = []
    lines.append("PROMPT INJECTION REFERENCE PACK")
    lines.append("")
    lines.append("Core defensive ideas:")
    for item in DEFENSE_PATTERNS[:max_defenses]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Most relevant techniques:")
    for item in techniques:
        lines.append(f"- {item['name']} [{item['family']}]")
        lines.append(f"  Summary: {item['summary']}")
        lines.append(f"  Signals: {', '.join(item['signals'][:3])}")
        lines.append(f"  Next-test ideas: {'; '.join(item['ideas'][:2])}")
    lines.append("")
    lines.append("Dataset notes:")
    for item in DATASET_NOTES:
        lines.append(f"- {item['name']}: {item['summary']}")
    lines.append("")
    lines.append("Sources:")
    for src in sources:
        lines.append(f"- {src['name']} ({src['published']}): {src['url']}")
        lines.append(f"  Notes: {src['notes']}")
    return "\n".join(lines)
