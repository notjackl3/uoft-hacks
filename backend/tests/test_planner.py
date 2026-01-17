from __future__ import annotations

import pytest

from app.models import PageFeature
from app.services.planner import build_planner_prompt, parse_planner_steps, PlannerError


def test_build_planner_prompt_includes_goal_url_and_features():
    prompt = build_planner_prompt(
        user_goal="buy wireless mouse under $30",
        initial_features=[
            PageFeature(index=0, type="input", text="", selector="#search", placeholder="Search", aria_label="Search"),
        ],
        url="https://amazon.com",
    )
    assert "buy wireless mouse under $30" in prompt
    assert "https://amazon.com" in prompt
    assert '"index": 0' in prompt
    assert '"selector": "#search"' in prompt


def test_parse_planner_steps_valid_json():
    raw = """
    {
      "steps": [
        {
          "step_number": 1,
          "action": "CLICK",
          "description": "Click search",
          "target_hints": {"type": "input", "text_contains": ["search"], "placeholder_contains": ["search"]},
          "expected_page_change": false
        }
      ]
    }
    """
    steps = parse_planner_steps(raw)
    assert len(steps) == 1
    assert steps[0].step_number == 1
    assert steps[0].action == "CLICK"


def test_parse_planner_steps_rejects_non_json():
    with pytest.raises(PlannerError):
        parse_planner_steps("hello not json")

