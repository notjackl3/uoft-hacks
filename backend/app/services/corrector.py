from __future__ import annotations

from typing import Optional

from app.models import PageFeature, PlannedStep, TargetHints
from app.utils.helpers import normalize_text


def _keywords_from_text(s: Optional[str]) -> list[str]:
    text = normalize_text(s)
    if not text:
        return []
    # split on whitespace, drop tiny tokens
    toks = [t for t in text.replace("/", " ").replace("-", " ").split() if len(t) >= 3]
    # dedupe preserve order
    out: list[str] = []
    for t in toks:
        if t not in out:
            out.append(t)
    return out[:5]


def update_hints_from_actual_feature(step: PlannedStep, actual: PageFeature) -> TargetHints:
    """
    Apply a lightweight "learning" update to target_hints based on the feature the user says is correct.
    """
    hints = step.target_hints.model_copy(deep=True)
    hints.type = actual.type

    # Prefer aria_label + text as match anchors
    text_keys = _keywords_from_text(actual.aria_label) + _keywords_from_text(actual.text)
    ph_keys = _keywords_from_text(actual.placeholder)

    for k in text_keys:
        if k not in hints.text_contains:
            hints.text_contains.append(k)
    for k in ph_keys:
        if k not in hints.placeholder_contains:
            hints.placeholder_contains.append(k)

    # Selector: store a safe substring if it's meaningful
    sel = actual.selector or ""
    if sel and not hints.selector_pattern:
        # Use substring pattern rather than regex to avoid invalid patterns
        hints.selector_pattern = sel[:64]

    # Role: we only have aria_label; keep as None unless provided
    return hints

