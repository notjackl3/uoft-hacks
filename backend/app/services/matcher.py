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


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein (edit) distance between two strings.
    Used for fuzzy text matching.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Calculate cost of insertions, deletions, and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def fuzzy_similarity(s1: str, s2: str) -> float:
    """
    Calculate fuzzy similarity between two strings (0.0 to 1.0).
    Uses Levenshtein distance normalized by the longer string length.
    
    Handles variations like:
    - "Sign in" vs "Sign In" vs "SIGN IN" (case insensitive)
    - "Create repository" vs "Create a new repository" (partial match)
    """
    n1 = normalize_text(s1).lower()
    n2 = normalize_text(s2).lower()
    
    if not n1 or not n2:
        return 0.0
    
    # Exact match after normalization
    if n1 == n2:
        return 1.0
    
    # One is substring of the other (handle "Create" vs "Create a new repository")
    if n1 in n2 or n2 in n1:
        # Score based on how much of the longer string is covered
        return min(len(n1), len(n2)) / max(len(n1), len(n2))
    
    # Levenshtein distance based similarity
    max_len = max(len(n1), len(n2))
    distance = levenshtein_distance(n1, n2)
    return max(0.0, 1.0 - (distance / max_len))


def _fuzzy_contains_any(haystack: str, needles: List[str], threshold: float = 0.7) -> tuple[bool, float]:
    """
    Check if haystack fuzzy-matches any needle.
    Returns (matched, best_score).
    
    Uses both exact substring matching AND fuzzy similarity.
    """
    h = normalize_text(haystack).lower()
    if not h:
        return False, 0.0
    
    best_score = 0.0
    for n in needles or []:
        n_norm = normalize_text(n).lower()
        if not n_norm:
            continue
        
        # Exact substring match (high confidence)
        if n_norm in h:
            return True, 1.0
        
        # Fuzzy match for each word in haystack
        h_words = h.split()
        for h_word in h_words:
            sim = fuzzy_similarity(h_word, n_norm)
            if sim > best_score:
                best_score = sim
        
        # Also check overall similarity
        overall_sim = fuzzy_similarity(h, n_norm)
        if overall_sim > best_score:
            best_score = overall_sim
    
    return best_score >= threshold, best_score

def _filter_features_for_step(current_step: PlannedStep, page_features: List[PageFeature]) -> List[PageFeature]:
    """
    Reduce candidate elements based on the step's action to lower confusion + shrink LLM fallback prompts.
    - CLICK: buttons + links
    - TYPE: inputs
    - otherwise: all
    If target_hints.type is present, restrict to that type.
    """
    if not page_features:
        return []

    # If hints specify a type, use that first.
    hinted_type = (current_step.target_hints.type or "").strip().lower()
    if hinted_type:
        filtered = [f for f in page_features if normalize_text(f.type) == normalize_text(hinted_type)]
        return filtered or page_features

    action = (current_step.action or "").upper()
    if action == "CLICK":
        filtered = [f for f in page_features if f.type in ("button", "link")]
        return filtered or page_features
    if action == "TYPE":
        filtered = [f for f in page_features if f.type == "input"]
        return filtered or page_features
    return page_features


def _contains_any(haystack: str, needles: List[str]) -> bool:
    """Legacy exact match - kept for backwards compatibility."""
    h = normalize_text(haystack)
    for n in needles or []:
        if normalize_text(n) and normalize_text(n) in h:
            return True
    return False


def _contains_any_fuzzy(haystack: str, needles: List[str]) -> tuple[bool, float]:
    """
    Enhanced matching that tries exact first, then falls back to fuzzy.
    Returns (matched, confidence_score).
    """
    # Try exact match first
    if _contains_any(haystack, needles):
        return True, 1.0
    
    # Fall back to fuzzy matching
    return _fuzzy_contains_any(haystack, needles, threshold=0.7)


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
    Match planned step to actual page feature using a weighted scoring algorithm
    with FUZZY TEXT MATCHING for better generalizability.
    
    Handles variations like:
    - "Sign in" vs "Sign In" vs "SIGN IN"
    - "Create repository" vs "Create a new repository"
    
    This does NOT call any LLM - it's purely algorithmic.
    """
    page_features = _filter_features_for_step(current_step, page_features)
    target_hints = current_step.target_hints
    best: Dict = {"matched": False, "feature_index": None, "confidence": 0.0, "feature": None}
    best_conf = 0.0

    for feature in page_features:
        score = 0.0
        max_score = 0.0

        # Type match
        if target_hints.type:
            max_score += WEIGHTS["type"]
            if normalize_text(feature.type) == normalize_text(target_hints.type):
                score += WEIGHTS["type"]

        # Text contains (feature.text OR aria_label) - FUZZY MATCHING
        if target_hints.text_contains:
            max_score += WEIGHTS["text"]
            combined = f"{feature.text or ''} {feature.aria_label or ''}"
            matched, fuzzy_score = _contains_any_fuzzy(combined, target_hints.text_contains)
            if matched:
                # Partial credit based on fuzzy score (minimum 50% if matched)
                score += WEIGHTS["text"] * max(0.5, fuzzy_score)

        # Placeholder contains - FUZZY MATCHING
        if target_hints.placeholder_contains:
            max_score += WEIGHTS["placeholder"]
            matched, fuzzy_score = _contains_any_fuzzy(feature.placeholder or "", target_hints.placeholder_contains)
            if matched:
                score += WEIGHTS["placeholder"] * max(0.5, fuzzy_score)

        # Role / aria-label (we only have aria_label in feature) - FUZZY MATCHING
        if target_hints.role:
            max_score += WEIGHTS["role"]
            role_sim = fuzzy_similarity(target_hints.role, feature.aria_label or "")
            if role_sim >= 0.7:
                score += WEIGHTS["role"] * role_sim

        # Selector pattern
        if target_hints.selector_pattern:
            max_score += WEIGHTS["selector"]
            if _selector_matches(feature.selector, target_hints.selector_pattern):
                score += WEIGHTS["selector"]

        confidence = (score / max_score) if max_score > 0 else 0.0
        if confidence > best_conf:
            best_conf = confidence
            best = {
                "matched": confidence > 0.4,  # Lowered threshold for fuzzy matching
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
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


async def fallback_to_gemini(current_step: PlannedStep, page_features: List[PageFeature]) -> Dict:
    """
    When algorithmic matching fails, ask OpenAI to choose the best feature index.
    Returns same format as match_element_to_step.
    
    Includes rate limiting and retry logic.
    """
    page_features = _filter_features_for_step(current_step, page_features)
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
    logger.info(f"Features count (filtered): {len(page_features)}")
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
