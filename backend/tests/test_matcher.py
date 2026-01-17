from __future__ import annotations

from app.models import PageFeature, PlannedStep, TargetHints
from app.services.matcher import match_element_to_step


def test_matcher_prefers_type_and_text():
    step = PlannedStep(
        step_number=1,
        action="CLICK",
        description="Click search",
        target_hints=TargetHints(type="input", text_contains=["search"], placeholder_contains=["search"]),
        expected_page_change=False,
    )
    features = [
        PageFeature(index=0, type="input", text="", selector="#q", placeholder="Search products", aria_label="Search"),
        PageFeature(index=1, type="button", text="Go", selector="#go", aria_label="Submit"),
    ]
    out = match_element_to_step(step, features)
    assert out["matched"] is True
    assert out["feature_index"] == 0
    assert out["confidence"] >= 0.8


def test_matcher_returns_no_match_for_empty_features():
    step = PlannedStep(step_number=1, action="CLICK", description="Click search", target_hints=TargetHints(type="input"))
    out = match_element_to_step(step, [])
    assert out["matched"] is False
    assert out["feature_index"] is None
    assert out["confidence"] == 0.0

