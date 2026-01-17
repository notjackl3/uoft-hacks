from __future__ import annotations

import json
import logging
from typing import List

from pydantic import ValidationError

from app.config import settings
from app.models import PageFeature, PlannedStep
from app.utils.helpers import JSONParseError, extract_json_object
from app.utils.rate_limiter import call_with_retry, RateLimitError

logger = logging.getLogger(__name__)


class PlannerError(RuntimeError):
    pass


def build_planner_prompt(user_goal: str, initial_features: list, url: str, page_title: str = "") -> str:
    features = [
        {
            "index": f.index,
            "type": f.type,
            "text": f.text or "",
            "placeholder": getattr(f, "placeholder", "") or "",
            "aria_label": getattr(f, "aria_label", "") or "",
            "href": getattr(f, "href", "") or "",
            "selector": f.selector or "",
        }
        for f in initial_features[:30]
    ]
    features_json = json.dumps(features, ensure_ascii=False)

    return f"""
You are a precise web automation planner.

GOAL: {user_goal}
PAGE_TITLE: {page_title}
URL: {url}
ELEMENTS_JSON: {features_json}

CRITICAL RULES:
- Use ONLY elements from ELEMENTS_JSON. Do NOT invent buttons/fields that aren't listed.
- Do NOT output a generic "search the web" plan unless the GOAL explicitly asks to search.
- If the current URL is unrelated to the goal (wrong site/tab), your first steps MUST be MANUAL (cannot be highlighted):
  - Use action=WAIT with expected_page_change=true and describe exactly what the user should do:
    1) Open a new tab
    2) Click the address bar
    3) Type the correct URL and press Enter
  - When you tell the user to go to a URL, you MUST include the full URL on its own line, so the user can copy and paste the link to use, or press on it to get to the correct site.
- Never propose creating/signing into accounts for unrelated services just because you're currently on their page (e.g., Google Account).
- When you reach the correct site, use ELEMENTS_JSON to decide what to do:
  - If elements suggest the user is already logged in (e.g. "Log out", "Settings", profile/account links), plan steps to log out first.
  - Otherwise plan steps to create a new account (sign up).
- Each step must be one atomic action: CLICK, TYPE, SCROLL, WAIT, DONE.
- Every CLICK/TYPE step MUST include target_hints with:
  - type (input/button/link)
  - AND at least one anchor: text_contains OR placeholder_contains OR selector_pattern.
- In target_hints, text_contains and placeholder_contains MUST be JSON arrays (use [] if none). Never use null.
- TYPE steps MUST include text_input.
  - If the goal requires unknown personal info (email/phone/password), use placeholders like "<EMAIL>", "<PASSWORD>" and say so in the description.
- End the plan with a DONE step.

OUTPUT:
Return JSON only, exactly:
{{
  "steps": [
    {{
      "step_number": 1,
      "action": "CLICK|TYPE|SCROLL|WAIT|DONE",
      "description": "...",
      "target_hints": {{
        "type": "input|button|link",
        "text_contains": ["..."],
        "placeholder_contains": ["..."],
        "selector_pattern": null,
        "role": null
      }},
      "text_input": null,
      "expected_page_change": false
    }}
  ]
}}
""".strip()

def parse_planner_steps(raw_text: str) -> List[PlannedStep]:
    try:
        data = extract_json_object(raw_text)
    except JSONParseError as e:
        raise PlannerError(f"Planner returned non-JSON output: {e}") from e

    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        raise PlannerError("Planner output missing non-empty 'steps' list")

    parsed: List[PlannedStep] = []
    for i, step in enumerate(steps):
        try:
            parsed.append(PlannedStep.model_validate(step))
        except ValidationError as e:
            raise PlannerError(f"Invalid step at index {i}: {e}") from e

    parsed.sort(key=lambda s: s.step_number)
    return parsed


def _call_openai_sync(prompt: str) -> str:
    """
    Synchronous OpenAI call (wrapped by rate limiter).
    """
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Fast and cheap model
        messages=[
            {"role": "user", "content": prompt}
        ],
        # Deterministic + faster: keep output small and enforce JSON mode.
        temperature=0.1,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


async def generate_workflow_plan(
    user_goal: str, initial_features: List[PageFeature], url: str, page_title: str = ""
) -> List[PlannedStep]:
    """
    Use OpenAI to generate complete step-by-step workflow.
    Includes rate limiting and retry logic.
    """
    prompt = build_planner_prompt(user_goal=user_goal, initial_features=initial_features, url=url, page_title=page_title)

    # Log what's being sent to OpenAI
    logger.info("=" * 60)
    logger.info("OPENAI PLANNER REQUEST")
    logger.info("=" * 60)
    logger.info(f"User Goal: {user_goal}")
    logger.info(f"URL: {url}")
    logger.info(f"Features count: {len(initial_features)}")
    logger.info(f"Prompt length: {len(prompt)} characters")
    logger.info("-" * 60)
    logger.info("FULL PROMPT:")
    logger.info(prompt)
    logger.info("=" * 60)

    try:
        # Use rate limiter with retry logic
        text = await call_with_retry(_call_openai_sync, prompt)
        
        # Log OpenAI's response
        logger.info("-" * 60)
        logger.info("OPENAI RESPONSE:")
        logger.info(text)
        logger.info("=" * 60)
        
        return parse_planner_steps(text)
    except RateLimitError as e:
        raise PlannerError(str(e)) from e
    except PlannerError:
        raise
    except Exception as e:  # pragma: no cover (SDK errors vary)
        logger.exception("OpenAI planner call failed")
        raise PlannerError(str(e)) from e
