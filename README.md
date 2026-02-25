# CompanyInsights.AI — Enterprise Hybrid Data Agent

An intelligent question-answering agent that routes natural language queries to the right data source: a **SQLite** financial database, a **ChromaDB** vector store of strategic documents, or a **hybrid** of both.

Powered by **GPT-4o-mini**, **FastAPI**, **React + Vite**, and **Tailwind CSS v4**.

---

## Architecture

```
User Question
     │
     ▼
 Router (LLM) ──► SQL Pipeline   ──► SQLite (ERP financials)
                ├► Vector Pipeline ──► ChromaDB (Q3 2025 Restructuring Memo)
                └► Hybrid Pipeline ──► Both, then synthesized by LLM
     │
     ▼
  Answer + Artifacts (SQL query, raw rows, retrieved chunks, relevance scores)
```

---

## Prerequisites

- **Python 3.9+** (3.11 recommended)
- **Node.js 18+** and **npm**
- An **OpenAI API key** with access to `gpt-4o-mini` and `text-embedding-3-small`

---

## Quick Start

### 1. Clone / download the project

```bash
cd "Just in case"
```

### 2. Set up the backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate.bat     # Windows CMD
# .venv\Scripts\Activate.ps1     # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Start the backend server

```bash
# From the backend/ directory, with .venv active
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

On first startup the server will:
1. Create and seed the SQLite database (`data/enterprise.db`) with 16 quarters of ERP financial data across two departments (D_402 and D_517)
2. Embed and store the Q3 2025 Strategic Restructuring Memo into ChromaDB (`data/chroma_db/`) using OpenAI embeddings

Subsequent starts reuse existing data (idempotent).

Verify the backend is running:
```bash
curl http://localhost:8000/api/health
```

API docs are available at: http://localhost:8000/docs

### 4. Set up the frontend

```bash
cd ../frontend

# Install dependencies
npm install

# Configure environment (optional — the dev proxy handles API routing)
cp .env.example .env
```

### 5. Start the frontend dev server

```bash
npm run dev
```

Open http://localhost:5173 in your browser.

---

## Project Structure

```
Just in case/
├── backend/
│   ├── agent/
│   │   ├── hybrid_pipeline.py   # Runs SQL + Vector in parallel, synthesizes
│   │   ├── prompts.py           # All LLM prompt templates
│   │   ├── router.py            # Classifies question → SQL / VECTOR / HYBRID
│   │   ├── sql_pipeline.py      # NL → SQL → execute → synthesize
│   │   └── vector_pipeline.py   # NL → ChromaDB retrieval → synthesize
│   ├── db/
│   │   ├── sqlite_init.py       # Creates + seeds the SQLite database
│   │   └── vector_init.py       # Embeds + stores memo chunks in ChromaDB
│   ├── models/
│   │   └── schemas.py           # Pydantic request / response models
│   ├── config.py                # Pydantic-settings configuration
│   ├── main.py                  # FastAPI app, lifespan, endpoints
│   ├── requirements.txt
│   └── .env.example
│
└── frontend/
    ├── src/
    │   ├── api/
    │   │   └── client.ts         # Typed fetch wrapper with cancellation
    │   ├── components/
    │   │   ├── ArtifactsPanel.tsx # Collapsible SQL / vector artifact viewer
    │   │   ├── ChatInput.tsx      # Auto-resizing textarea with submit button
    │   │   ├── ChatWindow.tsx     # Scrollable message list + empty state
    │   │   ├── ErrorBoundary.tsx  # React error boundary
    │   │   ├── MessageBubble.tsx  # Individual message with pipeline badge
    │   │   └── PipelineBadge.tsx  # SQL / VECTOR / HYBRID badge
    │   ├── types/
    │   │   └── index.ts           # TypeScript interfaces mirroring backend schemas
    │   ├── App.tsx                # Root component, owns all state
    │   └── main.tsx               # React entry point
    ├── package.json
    └── .env.example
```

---

## Sample Questions

| Type | Example |
|------|---------|
| **SQL** | What was the total revenue for department D_402 in fiscal year 2025? |
| **SQL** | Which department had more full-time employees at the end of Q4 2025? |
| **SQL** | What were the restructuring charges for both departments in Q3 2025? |
| **VECTOR** | What is Project Phoenix and what are its expected outcomes? |
| **VECTOR** | What is the strategic rationale for the restructuring? |
| **VECTOR** | What risks did the company identify in the restructuring memo? |
| **HYBRID** | Why did D_402's EBITDA margin improve from Q3 to Q4 2025? |
| **HYBRID** | Did Project Phoenix achieve its financial targets? |

---

## Environment Variables

### `backend/.env`

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | — | Your OpenAI API key |
| `LLM_MODEL` | | `gpt-4o-mini` | Chat completion model |
| `EMBEDDING_MODEL` | | `text-embedding-3-small` | Embedding model |
| `DATABASE_PATH` | | `./data/enterprise.db` | SQLite database path |
| `CHROMA_PERSIST_DIR` | | `./data/chroma_db` | ChromaDB data directory |
| `CHROMA_COLLECTION_NAME` | | `enterprise_docs` | ChromaDB collection name |
| `API_HOST` | | `0.0.0.0` | Uvicorn bind host |
| `API_PORT` | | `8000` | Uvicorn bind port |
| `VECTOR_TOP_K` | | `3` | Chunks to retrieve per query |
| `SQL_MAX_ROWS` | | `50` | Max rows returned from SQL |

### `frontend/.env`

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_BASE_URL` | *(uses dev proxy)* | Override API base URL for production |

---

## Development

### Backend tests (manual)

```bash
cd backend
source .venv/bin/activate

# Test the router individually
python -m agent.router

# Test the SQL pipeline
python -m agent.sql_pipeline

# Test the vector pipeline
python -m agent.vector_pipeline

# Re-initialize the vector store
python -m db.vector_init
```

### Frontend build

```bash
cd frontend
npm run build      # TypeScript compile + Vite bundle → dist/
npm run preview    # Preview the production build locally
npm run lint       # ESLint
```

---

## Production Deployment Notes

1. Set `CORS_ORIGINS` in `backend/.env` to your frontend domain
2. Set `VITE_API_BASE_URL` in `frontend/.env` to your backend URL
3. Serve the frontend `dist/` with nginx or a CDN
4. Run the backend with `uvicorn main:app --workers 2` (no `--reload`)
5. Mount `backend/data/` on persistent storage (SQLite + ChromaDB files)
