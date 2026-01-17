from __future__ import annotations

import json
from typing import List, Optional

from app.config import settings
from app.models import PageFeature, PlannedStep, TargetHints
from app.utils.helpers import extract_json_object


def _call_openai_sync(prompt: str) -> str:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def build_step_selector_prompt(
    canonical_goal: str,
    url: str,
    page_title: str,
    page_features: List[PageFeature],
    recent_steps: Optional[List[PlannedStep]] = None,
) -> str:
    features = [
        {
            "index": f.index,
            "type": f.type,
            "text": f.text or "",
            "placeholder": f.placeholder or "",
            "aria_label": f.aria_label or "",
            "href": f.href or "",
            "selector": f.selector or "",
            "value_len": int(getattr(f, "value_len", 0) or 0),
        }
        for f in (page_features or [])[:60]
    ]
    recent = [
        {
            "step_number": s.step_number,
            "action": s.action,
            "description": s.description,
            "text_input": s.text_input,
            "expected_page_change": s.expected_page_change,
            "target_hints": s.target_hints.model_dump() if s.target_hints else {},
        }
        for s in (recent_steps or [])[-5:]
    ]

    return f"""
You are selecting the SINGLE next step for a web tutoring assistant.
The user is the one who clicks/types; you only provide clear guidance and element hints.

GOAL: {canonical_goal}
PAGE_TITLE: {page_title}
URL: {url}
ELEMENTS_JSON: {json.dumps(features, ensure_ascii=False)}
RECENT_STEPS_JSON: {json.dumps(recent, ensure_ascii=False)}

Rules:
- Output JSON only, one object (not an array).
- action must be one of: CLICK, TYPE, SCROLL, WAIT, DONE
- If you need the user to use the browser UI (new tab/address bar), use action=WAIT with expected_page_change=true and include the full URL on its own line.
- If an input element already looks filled (value_len > 0), DO NOT suggest TYPE into it again unless the goal is to replace/edit it.
- For CLICK/TYPE you MUST include target_hints.type and at least one of:
  - text_contains (array, [] if none)
  - placeholder_contains (array, [] if none)
  - selector_pattern (string or null)
- TYPE must include text_input (or "<EMAIL>", "<PASSWORD>", etc)
- End with DONE only when the goal is achieved.

Output format:
{{
  "action": "CLICK|TYPE|SCROLL|WAIT|DONE",
  "description": "...",
  "target_hints": {{
    "type": "input|button|link",
    "text_contains": [],
    "placeholder_contains": [],
    "selector_pattern": null,
    "role": null
  }},
  "text_input": null,
  "expected_page_change": false
}}
""".strip()


async def select_next_step(
    step_number: int,
    canonical_goal: str,
    url: str,
    page_title: str,
    page_features: List[PageFeature],
    recent_steps: Optional[List[PlannedStep]] = None,
) -> PlannedStep:
    prompt = build_step_selector_prompt(
        canonical_goal=canonical_goal,
        url=url,
        page_title=page_title,
        page_features=page_features,
        recent_steps=recent_steps,
    )
    raw = _call_openai_sync(prompt)
    data = extract_json_object(raw)

    # Make it tolerant: fill defaults
    action = (data.get("action") or "WAIT").upper()
    description = data.get("description") or ""
    th = data.get("target_hints") or {}
    target_hints = TargetHints.model_validate(th) if isinstance(th, dict) else TargetHints()

    step = PlannedStep(
        step_number=step_number,
        action=action,  # type: ignore[arg-type]
        description=str(description),
        target_hints=target_hints,
        text_input=data.get("text_input"),
        expected_page_change=bool(data.get("expected_page_change", False)),
    )
    return step

