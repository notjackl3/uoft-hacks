from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from app.config import settings
from app.models import PageFeature, PlannedStep
from app.utils.helpers import extract_json_object, normalize_text

logger = logging.getLogger(__name__)


class MatcherError(RuntimeError):
    pass


WEIGHTS = {
    "type": 30,
    "text": 25,
    "placeholder": 20,
    "role": 15,
    "selector": 10,
}


def _contains_any(haystack: str, needles: List[str]) -> bool:
    h = normalize_text(haystack)
    for n in needles or []:
        if normalize_text(n) and normalize_text(n) in h:
            return True
    return False


def _selector_matches(selector: str, pattern: Optional[str]) -> bool:
    if not pattern:
        return False
    try:
        return re.search(pattern, selector or "", flags=re.IGNORECASE) is not None
    except re.error:
        # Treat invalid regex as plain substring
        return normalize_text(pattern) in normalize_text(selector)


def match_element_to_step(current_step: PlannedStep, page_features: List[PageFeature]) -> Dict:
    """
    Match planned step to actual page feature using a simple weighted scoring algorithm.
    """
    target_hints = current_step.target_hints
    best: Dict = {"matched": False, "feature_index": None, "confidence": 0.0, "feature": None}
    best_conf = 0.0

    for feature in page_features:
        score = 0
        max_score = 0

        # Type match
        if target_hints.type:
            max_score += WEIGHTS["type"]
            if normalize_text(feature.type) == normalize_text(target_hints.type):
                score += WEIGHTS["type"]

        # Text contains (feature.text OR aria_label)
        if target_hints.text_contains:
            max_score += WEIGHTS["text"]
            combined = f"{feature.text or ''} {feature.aria_label or ''}"
            if _contains_any(combined, target_hints.text_contains):
                score += WEIGHTS["text"]

        # Placeholder contains
        if target_hints.placeholder_contains:
            max_score += WEIGHTS["placeholder"]
            if _contains_any(feature.placeholder or "", target_hints.placeholder_contains):
                score += WEIGHTS["placeholder"]

        # Role / aria-label (we only have aria_label in feature)
        if target_hints.role:
            max_score += WEIGHTS["role"]
            if normalize_text(target_hints.role) in normalize_text(feature.aria_label):
                score += WEIGHTS["role"]

        # Selector pattern
        if target_hints.selector_pattern:
            max_score += WEIGHTS["selector"]
            if _selector_matches(feature.selector, target_hints.selector_pattern):
                score += WEIGHTS["selector"]

        confidence = (score / max_score) if max_score > 0 else 0.0
        if confidence > best_conf:
            best_conf = confidence
            best = {
                "matched": confidence > 0.5,
                "feature_index": feature.index,
                "confidence": float(confidence),
                "feature": feature,
            }

    return best


async def fallback_to_gemini(current_step: PlannedStep, page_features: List[PageFeature]) -> Dict:
    """
    When algorithmic matching fails, ask Gemini to choose the best feature index.
    Returns same format as match_element_to_step.
    """
    if not page_features:
        return {"matched": False, "feature_index": None, "confidence": 0.0, "feature": None}

    features_compact = [
        {
            "index": f.index,
            "type": f.type,
            "text": f.text or "",
            "placeholder": f.placeholder or "",
            "aria_label": f.aria_label or "",
            "selector": f.selector,
            "href": f.href or "",
        }
        for f in page_features
    ]

    prompt = f"""
You are matching a planned UI action to the correct element on a web page.

PLANNED STEP:
{current_step.model_dump()}

PAGE FEATURES (JSON list):
{features_compact}

TASK:
Pick the single best feature "index" to target for this step.

RULES:
- Output JSON ONLY
- If none match, return index=null and confidence=0

OUTPUT:
{{
  "index": 3,
  "confidence": 0.85
}}
""".strip()

    try:
        # Lazy import: keeps unit tests runnable without the SDK / sandbox cert issues.
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        data = extract_json_object(getattr(resp, "text", "") or "")
        idx = data.get("index")
        conf = float(data.get("confidence", 0.0) or 0.0)
        if idx is None:
            return {"matched": False, "feature_index": None, "confidence": 0.0, "feature": None}

        chosen = next((f for f in page_features if f.index == int(idx)), None)
        if chosen is None:
            return {"matched": False, "feature_index": None, "confidence": 0.0, "feature": None}
        return {"matched": conf > 0.5, "feature_index": chosen.index, "confidence": conf, "feature": chosen}
    except Exception as e:  # pragma: no cover
        logger.exception("Gemini fallback matcher failed")
        raise MatcherError(str(e)) from e

