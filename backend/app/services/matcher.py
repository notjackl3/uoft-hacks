from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from app.config import settings
from app.models import PageFeature, PlannedStep
from app.utils.helpers import extract_json_object, normalize_text
from app.utils.rate_limiter import call_with_retry, RateLimitError

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
    This does NOT call any LLM - it's purely algorithmic.
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


def _call_openai_matcher_sync(prompt: str) -> str:
    """
    Synchronous OpenAI call for matching (wrapped by rate limiter).
    """
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Fast and cheap model
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Lower temperature for more deterministic matching
    )
    return response.choices[0].message.content or ""


async def fallback_to_openai(current_step: PlannedStep, page_features: List[PageFeature]) -> Dict:
    """
    When algorithmic matching fails, ask OpenAI to choose the best feature index.
    Returns same format as match_element_to_step.
    
    Includes rate limiting and retry logic.
    """
    if not page_features:
        return {"matched": False, "feature_index": None, "confidence": 0.0, "feature": None}

    # Ultra-compact format
    elements = " | ".join([f"{f.index}:{f.type[0]}:{f.text or '-'}" for f in page_features[:15]])
    text_part = f" text:{current_step.text_input}" if current_step.text_input else ""

    prompt = f"""Match: {current_step.action} {current_step.description}{text_part}
Elements: {elements}
Reply JSON: {{"index":N,"confidence":0.9}} or {{"index":null,"confidence":0}}"""

    # Log what's being sent to OpenAI
    logger.info("=" * 60)
    logger.info("OPENAI MATCHER FALLBACK REQUEST")
    logger.info("=" * 60)
    logger.info(f"Step: {current_step.action} - {current_step.description}")
    logger.info(f"Features count: {len(page_features)}")
    logger.info(f"Prompt length: {len(prompt)} characters")
    logger.info("-" * 60)
    logger.info("FULL PROMPT:")
    logger.info(prompt)
    logger.info("=" * 60)

    try:
        # Use rate limiter with retry logic
        text = await call_with_retry(_call_openai_matcher_sync, prompt)
        
        # Log OpenAI's response
        logger.info("-" * 60)
        logger.info("OPENAI MATCHER RESPONSE:")
        logger.info(text)
        logger.info("=" * 60)
        
        data = extract_json_object(text)
        idx = data.get("index")
        conf = float(data.get("confidence", 0.0) or 0.0)
        
        if idx is None:
            return {"matched": False, "feature_index": None, "confidence": 0.0, "feature": None}

        chosen = next((f for f in page_features if f.index == int(idx)), None)
        if chosen is None:
            return {"matched": False, "feature_index": None, "confidence": 0.0, "feature": None}
        return {"matched": conf > 0.5, "feature_index": chosen.index, "confidence": conf, "feature": chosen}
    except RateLimitError as e:
        raise MatcherError(str(e)) from e
    except Exception as e:  # pragma: no cover
        logger.exception("OpenAI fallback matcher failed")
        raise MatcherError(str(e)) from e


# Alias for backwards compatibility
fallback_to_gemini = fallback_to_openai
