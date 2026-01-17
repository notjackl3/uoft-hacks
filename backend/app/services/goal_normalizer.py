from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from app.config import settings
from app.utils.helpers import extract_json_object


@dataclass(frozen=True)
class NormalizedGoal:
    raw_goal: str
    canonical_goal: str
    target_url: Optional[str]
    target_domain: Optional[str]
    task_type: str  # e.g. "create_account", "search", "checkout", "unknown"


def _domain_from_url(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _extract_url(text: str) -> Optional[str]:
    # Grab first http(s) URL if present
    m = re.search(r"(https?://[^\s<>'\"]+)", text or "", flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def infer_target_from_goal(raw_goal: str) -> NormalizedGoal:
    """
    Cheap heuristic inference to save credits.
    If uncertain, returns target_domain=None so caller can optionally use LLM normalization.
    """
    g = (raw_goal or "").strip()
    gl = g.lower()

    url = _extract_url(g)
    if url:
        dom = _domain_from_url(url)
        return NormalizedGoal(
            raw_goal=g,
            canonical_goal=g,
            target_url=url,
            target_domain=dom or None,
            task_type="unknown",
        )

    # Keyword -> canonical target
    if "instagram" in gl:
        return NormalizedGoal(
            raw_goal=g,
            canonical_goal="Create a new Instagram account",
            target_url="https://www.instagram.com/accounts/emailsignup/",
            target_domain="www.instagram.com",
            task_type="create_account",
        )

    if "amazon" in gl:
        # Use a representative URL, but allow any amazon.* domain match in the router.
        return NormalizedGoal(
            raw_goal=g,
            canonical_goal="Go to Amazon",
            target_url="https://www.amazon.com/",
            target_domain="www.amazon.com",
            task_type="unknown",
        )

    # Unknown: force LLM normalization if needed
    return NormalizedGoal(raw_goal=g, canonical_goal=g, target_url=None, target_domain=None, task_type="unknown")


def _call_openai_sync(prompt: str) -> str:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


async def normalize_goal_llm(raw_goal: str) -> NormalizedGoal:
    """
    LLM normalization with strict JSON output.
    """
    prompt = f"""
You are normalizing a user's goal for a web-assistant.

USER_GOAL: {raw_goal}

Return JSON only with:
{{
  "canonical_goal": "...",
  "task_type": "create_account|search|checkout|unknown",
  "target_url": "https://..." or null,
  "target_domain": "example.com" or null
}}

Rules:
- If the user mentions a site name (e.g. Instagram), set target_url and target_domain.
- target_domain must be just the hostname (no scheme).
""".strip()

    # Keep this sync call wrapped outside; caller can rate-limit/retry if desired.
    text = _call_openai_sync(prompt)
    data = extract_json_object(text)
    target_url = data.get("target_url")
    if target_url is not None and not isinstance(target_url, str):
        target_url = None
    target_domain = data.get("target_domain")
    if target_domain is not None and not isinstance(target_domain, str):
        target_domain = None
    canonical_goal = data.get("canonical_goal") or raw_goal
    task_type = data.get("task_type") or "unknown"
    return NormalizedGoal(
        raw_goal=raw_goal,
        canonical_goal=str(canonical_goal),
        target_url=target_url,
        target_domain=str(target_domain).lower() if target_domain else None,
        task_type=str(task_type),
    )

