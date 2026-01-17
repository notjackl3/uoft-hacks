from __future__ import annotations

import json
import re
from typing import Any, Dict


class JSONParseError(ValueError):
    pass


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from an LLM response that may include code fences or extra text.
    """
    if not text:
        raise JSONParseError("Empty response")

    # Remove code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)

    # First try direct parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Fallback: find the first {...} block
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise JSONParseError("No JSON object found in response")

    try:
        data = json.loads(match.group(0))
    except Exception as e:
        raise JSONParseError(f"Invalid JSON: {e}") from e

    if not isinstance(data, dict):
        raise JSONParseError("JSON root is not an object")
    return data


def normalize_text(s: str | None) -> str:
    return (s or "").strip().lower()

