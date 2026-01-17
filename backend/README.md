## Universal On-Screen Tutor Agent — Backend

FastAPI backend for planning workflows (Gemini), matching UI steps to on-page elements, and tracking sessions in MongoDB.

### Setup

- **Python**: 3.11+
- **Install deps**:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment

This environment blocks committing dotfiles like `.env.example`, so use `backend/env.example` as your template:

```bash
cp env.example .env
```

Fill in `MONGODB_URI`, `GEMINI_API_KEY`, `VOYAGE_API_KEY`.

### Run

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Health: `GET /health`

### API

- `POST /api/session/start`
- `POST /api/session/next`
- `POST /api/session/correct`
- `GET /api/session/{session_id}/status`

### Tests

```bash
cd backend
pytest -q
```

### Mock scenario runner (see outputs without Gemini/Mongo)

This runs the API in-memory (fake DB), mocks the **goal embedding** + **planner output**, and prints what `/start` and `/next` return for each **stage** (step number).

```bash
cd backend
python3 scripts/run_mock_scenario.py --scenario mock_scenarios/amazon_search.json
```

To jump to a later stage (mock the “stage of the query/workflow”):

```bash
cd backend
python3 scripts/run_mock_scenario.py --scenario mock_scenarios/amazon_search.json --start-stage 3
```

