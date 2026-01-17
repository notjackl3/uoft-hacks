from __future__ import annotations

import httpx
import pytest

from app.database import get_db
from app.main import create_app
from app.models import PlannedStep, TargetHints


@pytest.mark.anyio
async def test_session_flow_start_next_correct_status(fake_db, monkeypatch):
    # Mock external services
    async def _embed_text(_text: str):
        return [0.0] * 8

    async def _normalize_goal_llm(_raw_goal: str):
        # Pretend we inferred the target domain from the goal so /start does not short-circuit.
        from app.services.goal_normalizer import NormalizedGoal

        return NormalizedGoal(
            raw_goal=_raw_goal,
            canonical_goal=_raw_goal,
            target_url=None,
            target_domain="example.com",
            task_type="unknown",
        )

    async def _select_next_step(step_number, canonical_goal, url, page_title, page_features, recent_steps=None):
        # Return deterministic steps in "single_step" mode.
        if step_number == 1:
            return PlannedStep(
                step_number=1,
                action="CLICK",
                description="Click the search bar",
                target_hints=TargetHints(type="input", text_contains=["search"], placeholder_contains=["search"]),
                expected_page_change=False,
            )
        if step_number == 2:
            return PlannedStep(
                step_number=2,
                action="TYPE",
                description="Type the query",
                target_hints=TargetHints(type="input", text_contains=["search"], placeholder_contains=["search"]),
                text_input="wireless mouse",
                expected_page_change=False,
            )
        return PlannedStep(step_number=step_number, action="DONE", description="Done", target_hints=TargetHints())

    import app.routes.session as session_routes

    monkeypatch.setattr(session_routes, "embed_text", _embed_text)
    monkeypatch.setattr(session_routes, "normalize_goal_llm", _normalize_goal_llm)
    monkeypatch.setattr(session_routes, "select_next_step", _select_next_step)

    app = create_app(with_db=False)
    app.dependency_overrides[get_db] = lambda: fake_db
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        start = await client.post(
            "/api/session/start",
            json={
                "user_goal": "buy wireless mouse under $30",
                "initial_page_features": [
                    {
                        "index": 0,
                        "type": "input",
                        "text": "",
                        "selector": "#search",
                        "placeholder": "Search Amazon",
                        "aria_label": "Search",
                    },
                    {"index": 1, "type": "button", "text": "Go", "selector": "#go"},
                ],
                "url": "https://example.com",
                "page_title": "Example",
            },
        )
        assert start.status_code == 200
        body = start.json()
        assert "session_id" in body
        assert body["first_step"]["target_feature_index"] == 0

        session_id = body["session_id"]

        nxt = await client.post(
            "/api/session/next",
            json={
                "session_id": session_id,
                "page_features": [
                    {
                        "index": 0,
                        "type": "input",
                        "text": "",
                        "selector": "#search",
                        "placeholder": "Search Amazon",
                        "aria_label": "Search",
                    },
                    {"index": 1, "type": "button", "text": "Go", "selector": "#go"},
                ],
                "previous_action_result": {"success": True},
            },
        )
        assert nxt.status_code == 200
        nxt_body = nxt.json()
        assert nxt_body["step_number"] == 2
        assert nxt_body["action"] == "TYPE"
        assert nxt_body["target_feature_index"] == 0
        assert nxt_body["text_input"] == "wireless mouse"

        corr = await client.post(
            "/api/session/correct",
            json={"session_id": session_id, "feedback": "wrong_element", "actual_feature_index": 1},
        )
        assert corr.status_code == 200
        corr_body = corr.json()
        assert corr_body["ok"] is True
        assert corr_body["updated_target_hints"]["type"] == "button"

        st = await client.get(f"/api/session/{session_id}/status")
        assert st.status_code == 200
        st_body = st.json()
        assert st_body["session_id"] == session_id
        assert st_body["status"] == "in_progress"

