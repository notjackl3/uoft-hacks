from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import anyio
import httpx

# Ensure `app.*` imports work when running as a script.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database import get_db
from app.main import create_app
from app.models import PlannedStep
from app.utils.inmemory_db import FakeDB


async def run_scenario(path: Path, start_stage: int) -> None:
    scenario = json.loads(path.read_text())

    planned_steps = [PlannedStep.model_validate(s) for s in scenario["planned_steps"]]

    # Mock external services by patching the route module functions directly.
    import app.routes.session as session_routes

    async def _embed_text(_text: str):
        return [0.0] * 8

    async def _generate_workflow_plan(user_goal: str, initial_features, url: str):
        return planned_steps

    session_routes.embed_text = _embed_text  # type: ignore[assignment]
    session_routes.generate_workflow_plan = _generate_workflow_plan  # type: ignore[assignment]

    fake_db = FakeDB()
    app = create_app(with_db=False)
    app.dependency_overrides[get_db] = lambda: fake_db

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Stage 1 start payload uses the stage-1 features.
        stage1 = next(s for s in scenario["stages"] if int(s["stage"]) == 1)
        start_resp = await client.post(
            "/api/session/start",
            json={
                "user_goal": scenario["user_goal"],
                "initial_page_features": stage1["page_features"],
                "url": scenario["url"],
                "page_title": scenario.get("page_title", ""),
            },
        )
        print("\n=== /start ===")
        print(json.dumps(start_resp.json(), indent=2))
        start_resp.raise_for_status()

        session_id = start_resp.json()["session_id"]

        # If caller wants to jump to a later stage, adjust session counters.
        if start_stage > 1:
            # Find the session doc and patch it (FakeDB stores docs in memory).
            doc = await fake_db.sessions.find_one({"session_id": session_id})
            if not doc:
                raise RuntimeError("session doc missing")
            # Set current to requested stage, but last_sent to stage-1 so /next does not auto-increment.
            await fake_db.sessions.update_one(
                {"session_id": session_id},
                {"$set": {"current_step_number": start_stage, "last_sent_step_number": start_stage - 1}},
            )

        # Now iterate stages >= start_stage and call /next using those page features.
        for st in sorted(scenario["stages"], key=lambda x: int(x["stage"])):
            stage_num = int(st["stage"])
            if stage_num < start_stage:
                continue

            next_resp = await client.post(
                "/api/session/next",
                json={
                    "session_id": session_id,
                    "page_features": st["page_features"],
                    "previous_action_result": {"success": True},
                },
            )
            print(f"\n=== /next (stage={stage_num}) ===")
            print(json.dumps(next_resp.json(), indent=2))
            next_resp.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a mock scenario through the backend and print JSON outputs.")
    parser.add_argument(
        "--scenario",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "mock_scenarios" / "amazon_search.json"),
        help="Path to a scenario JSON file",
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        help="Which planned step/stage number to start from (mocks the stage of the query/workflow).",
    )
    args = parser.parse_args()
    anyio.run(run_scenario, Path(args.scenario), int(args.start_stage))


if __name__ == "__main__":
    main()

