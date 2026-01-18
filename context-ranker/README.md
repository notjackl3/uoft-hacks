# Context Ranker

A Node.js (TypeScript) backend service that stores, indexes, and retrieves ranked context passages for browser automation agents. Designed to work with backboard.io.

## Features

- **Ingestion Pipeline**: Crawl → Extract → Chunk → Embed → Store
- **Context Retrieval**: Semantic search + heuristic reranking
- **XGBoost Reranking**: Optional ML-based reranking (Python)
- **Feedback Logging**: Collect training data for model improvement
- **MongoDB Storage**: Pages, passages, and training events

## Quick Start

### 1. Prerequisites

- Node.js 18+
- Docker (for local MongoDB)
- Python 3.8+ (for XGBoost, optional)

### 2. Setup

```bash
# Clone and navigate
cd context-ranker

# Start MongoDB (Docker)
docker-compose up -d

# Install dependencies
npm install

# Copy environment file
cp .env.example .env
```

### 3. Configure Environment

Edit `.env`:

```env
# MongoDB
MONGODB_URI=mongodb://localhost:27017/context-ranker

# Embeddings (OpenAI or Backboard)
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=sk-your-openai-key

# Optional: Use Backboard embeddings
# EMBEDDING_PROVIDER=backboard
# BACKBOARD_BASE_URL=https://api.backboard.io

# XGBoost (optional)
USE_XGBOOST=false
```

### 4. Start the Server

```bash
# Development (with hot reload)
npm run dev

# Production
npm run build
npm start
```

Server runs at `http://localhost:3001`

## API Endpoints

### POST /context

Get ranked context passages for a task.

**Request:**
```json
{
  "site": "stripe.com",
  "task": "How do I accept payments?",
  "currentUrl": "https://stripe.com/docs",
  "visibleTextHints": "Dashboard overview..."
}
```

**Response:**
```json
{
  "passages": [
    {
      "url": "https://stripe.com/docs/payments",
      "title": "Accept a payment",
      "content": "Use the Payment Intents API...",
      "score": 0.823,
      "features": {
        "hasSteps": true,
        "hasCode": true,
        "hasUIWords": false,
        "authorityTier": 1
      }
    }
  ],
  "query": "How do I accept payments?",
  "candidatesScanned": 485,
  "timeTakenMs": 279
}
```

### POST /feedback

Log outcomes for training data.

**Request:**
```json
{
  "sessionId": "session-123",
  "task": "How do I accept payments?",
  "outcome": "success",
  "chosenPassageId": "abc123",
  "actionsTaken": 5,
  "timeSpentMs": 12000
}
```

### GET /feedback/export

Export training events for XGBoost.

```bash
curl http://localhost:3001/feedback/export?limit=1000
```

### GET /health

Health check endpoint.

## Ingestion

### Ingest a Website

```bash
# Basic usage
npm run ingest -- https://stripe.com/docs 50

# Arguments: <url> <maxPages>
```

The crawler:
- Stays within the same domain
- Prioritizes /docs, /help, /api, /guides paths
- Extracts main content (removes nav/footer)
- Chunks into 300-450 token passages
- Computes embeddings
- Stores with features (hasSteps, hasCode, etc.)

## XGBoost Reranking (Optional)

### Install Python Dependencies

```bash
pip install xgboost pymongo numpy
```

### Train the Model

```bash
# Ensure you have feedback data first
python scripts/train_xgboost.py
```

This creates `models/ranker.json`.

### Enable XGBoost

Set in `.env`:
```env
USE_XGBOOST=true
```

The server will use XGBoost scores combined with semantic similarity.

## Architecture

```
context-ranker/
├── src/
│   ├── index.ts              # Express server entry
│   ├── api/
│   │   ├── context.ts        # POST /context
│   │   └── feedback.ts       # POST /feedback
│   ├── ingestion/
│   │   ├── crawler.ts        # Domain-limited crawler
│   │   ├── extractor.ts      # Content extraction
│   │   ├── chunker.ts        # Text chunking
│   │   └── pipeline.ts       # Orchestration
│   ├── services/
│   │   ├── database.ts       # MongoDB connection
│   │   └── embeddings.ts     # Embedding providers
│   ├── models/
│   │   └── schemas.ts        # TypeScript interfaces
│   ├── xgboost/
│   │   └── inference.ts      # XGBoost integration
│   └── utils/
│       └── config.ts         # Environment config
├── scripts/
│   ├── train_xgboost.py      # XGBoost training
│   └── infer_xgboost.py      # XGBoost inference
├── models/                    # XGBoost model files
├── docker-compose.yml         # Local MongoDB
└── package.json
```

## MongoDB Collections

### pages
Crawled page metadata.
```typescript
{
  url: string;
  domain: string;
  path: string;
  title: string;
  headings: string[];
  crawledAt: Date;
  contentHash: string;
  passageCount: number;
}
```

### passages
Chunked content with embeddings.
```typescript
{
  pageId: ObjectId;
  url: string;
  content: string;
  embedding: number[];  // 1536 dimensions (OpenAI)
  features: {
    hasSteps: boolean;
    hasCode: boolean;
    hasUIWords: boolean;
    marketingScore: number;
    authorityTier: number;  // 1=docs, 2=guides, 3=other
    freshnessDays: number;
    chunkPosition: number;
    tokenCount: number;
  }
}
```

### training_events
Feedback for XGBoost training.
```typescript
{
  sessionId: string;
  task: string;
  outcome: 'success' | 'failure' | 'partial' | 'abandoned';
  chosenPassageId?: string;
  candidatePassageIds: string[];
  chosenFeatures?: PassageFeatures;
  candidateFeatures?: Array<{ passageId: string; features: PassageFeatures }>;
}
```

## Ranking Algorithm

### Heuristic Weights (default)

| Feature | Weight |
|---------|--------|
| Semantic similarity | 50% |
| hasUIWords | 12% |
| authorityTier | 10% |
| hasSteps | 10% |
| hasCode | 8% |
| freshness | 5% |
| marketingPenalty | -5% |

### XGBoost Mode

When `USE_XGBOOST=true`:
- 60% XGBoost score
- 40% Semantic similarity

## Development

```bash
# Run tests
npm test

# Type check
npm run typecheck

# Lint
npm run lint
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| MONGODB_URI | Yes | - | MongoDB connection string |
| EMBEDDING_API_KEY | Yes | - | OpenAI or Backboard API key |
| EMBEDDING_PROVIDER | No | openai | `openai` or `backboard` |
| EMBEDDING_MODEL | No | text-embedding-3-small | Embedding model name |
| USE_XGBOOST | No | false | Enable XGBoost reranking |
| PORT | No | 3001 | Server port |
| NODE_ENV | No | development | Environment |

## License

MIT
