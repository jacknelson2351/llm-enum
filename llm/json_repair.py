"""Extract JSON from LLM output that may contain markdown fences, preamble, or truncated output."""

from __future__ import annotations

import json
import re

# Match ```json ... ``` or ``` ... ``` blocks
_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)

# Find first [ or { and grab everything from there
_ARRAY_START_RE = re.compile(r"(\[.+)", re.DOTALL)
_OBJ_START_RE = re.compile(r"(\{.+)", re.DOTALL)


def _try_parse(text: str) -> dict | list | None:
    """Try json.loads, return None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _repair_truncated(text: str) -> dict | list | None:
    """Try to close truncated JSON by adding missing brackets/braces."""
    text = text.rstrip().rstrip(",")
    # Count unmatched openers
    opens = {"[": 0, "{": 0}
    closers = {"]": "[", "}": "{"}
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in opens:
            opens[ch] += 1
        elif ch in closers:
            opens[closers[ch]] = max(0, opens[closers[ch]] - 1)

    # Append missing closers in reverse order (} then ])
    suffix = "}" * opens["{"] + "]" * opens["["]
    if suffix:
        result = _try_parse(text + suffix)
        if result is not None:
            return result

    return None


def extract_json(raw: str) -> dict | list:
    """Parse JSON from LLM output, handling fences, preamble, and truncation.

    Tries in order:
    1. Direct json.loads
    2. Extract from markdown code fences
    3. Find first [ or { and parse from there
    4. Attempt truncation repair on all candidates
    5. Raise ValueError with context
    """
    text = raw.strip()
    if not text:
        raise ValueError("Empty response from model")

    # 1. Direct parse
    result = _try_parse(text)
    if result is not None:
        return result

    # 2. Code fences — try all matches (some models emit multiple)
    for fence_match in _FENCE_RE.finditer(text):
        inner = fence_match.group(1).strip()
        result = _try_parse(inner)
        if result is not None:
            return result
        # Try repairing truncated fence content
        result = _repair_truncated(inner)
        if result is not None:
            return result

    # 3. Find first JSON structure in the text
    candidates = []
    for pattern in (_ARRAY_START_RE, _OBJ_START_RE):
        m = pattern.search(text)
        if m:
            candidates.append(m.group(1).strip())

    for candidate in candidates:
        result = _try_parse(candidate)
        if result is not None:
            return result

    # 4. Repair truncated candidates
    # Also try stripping trailing fence markers that weren't matched
    for candidate in candidates:
        clean = candidate.rstrip("`").rstrip()
        result = _repair_truncated(clean)
        if result is not None:
            return result

    # 5. Nothing worked
    preview = text[:160].replace("\n", " ")
    raise ValueError(f"Could not extract JSON from model output: {preview}...")
