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

    async def _generate_workflow_plan(user_goal, initial_features, url):
        return [
            PlannedStep(
                step_number=1,
                action="CLICK",
                description="Click the search bar",
                target_hints=TargetHints(type="input", text_contains=["search"], placeholder_contains=["search"]),
                expected_page_change=False,
            ),
            PlannedStep(
                step_number=2,
                action="TYPE",
                description="Type the query",
                target_hints=TargetHints(type="input", text_contains=["search"], placeholder_contains=["search"]),
                text_input="wireless mouse",
                expected_page_change=False,
            ),
            PlannedStep(
                step_number=3,
                action="CLICK",
                description="Click Go",
                target_hints=TargetHints(type="button", text_contains=["go"]),
                expected_page_change=True,
            ),
            PlannedStep(step_number=4, action="DONE", description="Done", target_hints=TargetHints()),
        ]

    import app.routes.session as session_routes

    monkeypatch.setattr(session_routes, "embed_text", _embed_text)
    monkeypatch.setattr(session_routes, "generate_workflow_plan", _generate_workflow_plan)

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

