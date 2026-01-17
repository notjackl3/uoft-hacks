"""
Unified Planner - Combines goal normalization + first step + task outline in ONE LLM call.

This reduces latency by eliminating 2 separate LLM calls at session start:
- Previously: normalize_goal_llm() + select_next_step() = 2 calls
- Now: unified_plan() = 1 call

The output includes:
1. canonical_goal - Normalized version of the user's goal
2. target_url/target_domain - Where the user needs to go (if applicable)
3. task_outline - High-level phases of the task (for progress tracking)
4. first_step - The immediate next action to take
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse

from app.config import settings
from app.models import PageFeature, PlannedStep, TargetHints
from app.utils.helpers import extract_json_object

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnifiedPlanResult:
    """Result from unified planning - combines normalization + outline + first step"""
    canonical_goal: str
    target_url: Optional[str]
    target_domain: Optional[str]
    task_type: str  # e.g. "create_account", "search", "navigate", "form_fill", "unknown"
    task_outline: List[str]  # High-level phases like ["Navigate to site", "Fill form", "Submit"]
    current_phase: int  # 0-indexed phase we're currently on
    first_step: PlannedStep


def _domain_from_url(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _call_openai_sync(prompt: str) -> str:
    """Synchronous OpenAI call for unified planning."""
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def build_unified_prompt(
    user_goal: str,
    url: str,
    page_title: str,
    page_features: List[PageFeature],
) -> str:
    """Build the prompt for unified planning."""
    features_data = [
        {
            "index": f.index,
            "type": f.type,
            "text": f.text or "",
            "placeholder": f.placeholder or "",
            "aria_label": f.aria_label or "",
            "href": f.href or "",
            "value_len": int(getattr(f, "value_len", 0) or 0),
        }
        for f in (page_features or [])[:30]
    ]

    return f"""
You are a web assistant planner. Given a user goal, create a UNIFIED PLAN with:
1. Normalized goal understanding
2. High-level task outline (phases)
3. The immediate first step to take

USER_GOAL: {user_goal}
CURRENT_URL: {url}
PAGE_TITLE: {page_title}
PAGE_ELEMENTS: {json.dumps(features_data, ensure_ascii=False)}

Output ONLY valid JSON with this structure:
{{
  "canonical_goal": "Clear, actionable version of the goal",
  "target_url": "https://..." or null (if user needs to navigate somewhere else),
  "target_domain": "example.com" or null,
  "task_type": "create_account|search|navigate|form_fill|purchase|settings|unknown",
  "task_outline": [
    "Phase 1: Brief description",
    "Phase 2: Brief description",
    ...
  ],
  "current_phase": 0,
  "first_step": {{
    "action": "CLICK|TYPE|SCROLL|WAIT|DONE",
    "description": "What the user should do",
    "target_hints": {{
      "type": "input|button|link",
      "text_contains": [],
      "placeholder_contains": [],
      "selector_pattern": null,
      "role": null
    }},
    "text_input": null or "text to type",
    "expected_page_change": false
  }}
}}

Rules:
- task_outline should have 2-6 phases covering the entire goal
- current_phase starts at 0
- If user is on wrong site, first_step should be WAIT with target_url in description
- For CLICK/TYPE, provide target_hints to match elements
- If goal appears already complete based on page state, use action=DONE
- Be concise but complete

Output valid JSON only, no markdown.
""".strip()


async def unified_plan(
    user_goal: str,
    url: str,
    page_title: str,
    page_features: List[PageFeature],
) -> UnifiedPlanResult:
    """
    Single LLM call that returns goal normalization + task outline + first step.
    
    This replaces the previous pattern of:
    1. normalize_goal_llm() 
    2. select_next_step()
    
    Reducing 2 LLM calls to 1.
    """
    prompt = build_unified_prompt(user_goal, url, page_title, page_features)
    
    logger.info("=" * 60)
    logger.info("UNIFIED PLANNER REQUEST")
    logger.info("=" * 60)
    logger.info(f"Goal: {user_goal}")
    logger.info(f"URL: {url}")
    logger.info(f"Features count: {len(page_features)}")
    
    raw = _call_openai_sync(prompt)
    
    logger.info("-" * 60)
    logger.info("UNIFIED PLANNER RESPONSE:")
    logger.info(raw[:500] + "..." if len(raw) > 500 else raw)
    logger.info("=" * 60)
    
    data = extract_json_object(raw)
    
    # Parse response with defaults
    canonical_goal = str(data.get("canonical_goal") or user_goal)
    target_url = data.get("target_url")
    if target_url is not None and not isinstance(target_url, str):
        target_url = None
    target_domain = data.get("target_domain")
    if target_domain is not None and not isinstance(target_domain, str):
        target_domain = None
    if target_domain:
        target_domain = target_domain.lower()
    
    task_type = str(data.get("task_type") or "unknown")
    task_outline = data.get("task_outline") or ["Complete the task"]
    if not isinstance(task_outline, list):
        task_outline = [str(task_outline)]
    task_outline = [str(phase) for phase in task_outline]
    
    current_phase = int(data.get("current_phase", 0) or 0)
    
    # Parse first step
    step_data = data.get("first_step") or {}
    action = (step_data.get("action") or "WAIT").upper()
    description = str(step_data.get("description") or "")
    
    th = step_data.get("target_hints") or {}
    target_hints = TargetHints.model_validate(th) if isinstance(th, dict) else TargetHints()
    
    first_step = PlannedStep(
        step_number=1,
        action=action,  # type: ignore[arg-type]
        description=description,
        target_hints=target_hints,
        text_input=step_data.get("text_input"),
        expected_page_change=bool(step_data.get("expected_page_change", False)),
    )
    
    return UnifiedPlanResult(
        canonical_goal=canonical_goal,
        target_url=target_url,
        target_domain=target_domain,
        task_type=task_type,
        task_outline=task_outline,
        current_phase=current_phase,
        first_step=first_step,
    )


def infer_target_fast(raw_goal: str) -> tuple[Optional[str], Optional[str]]:
    """
    Fast heuristic to detect target URL/domain without LLM.
    Returns (target_url, target_domain) or (None, None) if unknown.
    
    This is used as a pre-check before calling the unified planner,
    to handle obvious navigation cases cheaply.
    """
    g = (raw_goal or "").strip().lower()
    
    # Check for explicit URLs in the goal
    import re
    url_match = re.search(r"(https?://[^\s<>'\"]+)", raw_goal, flags=re.IGNORECASE)
    if url_match:
        url = url_match.group(1)
        domain = _domain_from_url(url)
        return url, domain or None
    
    # Common site keywords -> URLs
    site_mappings = {
        "github": ("https://github.com/", "github.com"),
        "instagram": ("https://www.instagram.com/", "www.instagram.com"),
        "twitter": ("https://twitter.com/", "twitter.com"),
        "x.com": ("https://x.com/", "x.com"),
        "facebook": ("https://www.facebook.com/", "www.facebook.com"),
        "linkedin": ("https://www.linkedin.com/", "www.linkedin.com"),
        "amazon": ("https://www.amazon.com/", "www.amazon.com"),
        "google": ("https://www.google.com/", "www.google.com"),
        "youtube": ("https://www.youtube.com/", "www.youtube.com"),
        "reddit": ("https://www.reddit.com/", "www.reddit.com"),
    }
    
    for keyword, (url, domain) in site_mappings.items():
        if keyword in g:
            return url, domain
    
    return None, None
