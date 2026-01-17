from __future__ import annotations

import logging
from typing import List

from pydantic import ValidationError

from app.config import settings
from app.models import PageFeature, PlannedStep
from app.utils.helpers import JSONParseError, extract_json_object

logger = logging.getLogger(__name__)


class PlannerError(RuntimeError):
    pass


def build_planner_prompt(user_goal: str, initial_features: List[PageFeature], url: str) -> str:
    feature_lines = []
    for f in initial_features:
        feature_lines.append(
            f"- index={f.index}, type={f.type}, text={f.text or ''}, placeholder={f.placeholder or ''}, "
            f"aria_label={f.aria_label or ''}, selector={f.selector}"
        )

    features_block = "\n".join(feature_lines) if feature_lines else "- (no features provided)"

    return f"""
You are planning a step-by-step workflow for an accessibility tutor.

USER GOAL: {user_goal}

STARTING PAGE:
- URL: {url}
- Available features:
{features_block}

YOUR TASK:
Generate a complete step-by-step plan.

RULES:
1. Break into atomic actions (CLICK, TYPE, SCROLL, WAIT, DONE)
2. Each step needs "target_hints" for element matching
3. Mark if step causes page change
4. Keep simple and clear
5. OUTPUT JSON ONLY (no markdown, no commentary)

OUTPUT FORMAT:
{{
  "steps": [
    {{
      "step_number": 1,
      "action": "CLICK",
      "description": "Click the search bar",
      "target_hints": {{
        "type": "input",
        "text_contains": ["search"],
        "placeholder_contains": ["search"],
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


async def generate_workflow_plan(user_goal: str, initial_features: List[PageFeature], url: str) -> List[PlannedStep]:
    """
    Use Gemini to generate complete step-by-step workflow.
    """
    prompt = build_planner_prompt(user_goal=user_goal, initial_features=initial_features, url=url)

    try:
        # Lazy import: keeps unit tests runnable without the SDK / sandbox cert issues.
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None) or ""
        return parse_planner_steps(text)
    except PlannerError:
        raise
    except Exception as e:  # pragma: no cover (SDK errors vary)
        logger.exception("Gemini planner call failed")
        raise PlannerError(str(e)) from e

