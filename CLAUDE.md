# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Acela AI Agent is a FastAPI service that answers questions about Celo (an Ethereum L2) using a LangGraph-based multi-path agent. The agent routes queries to a Pinecone vector store (RAG), Tavily web search, or direct LLM fallback, with quality-grading at retrieval and generation stages.

## Commands

### Setup and Running

```bash
# Install dependencies (uses uv)
uv sync

# Run dev server
uv run uvicorn app.main:server --reload

# Run with Docker
docker compose up --build
```

### Index Documents (required before first run)

```bash
python index_documents.py
```

Loads Celo docs from GitHub, chunks them, and indexes into Pinecone (`acela-test-index`).

## Architecture

### Request Flow

```
POST /chat
  → agent_graph.py (LangGraph StateGraph)
    → route_question (router.py) — classifies intent
      → vectorstore path: retrieve → grade_documents → generate → hallucination_grade
      → web_search path: tavily search → generate
      → llm_fallback path: direct LLM response
```

### Key Files

- `app/main.py` — FastAPI app (`server`), `/chat` endpoint, CORS config
- `app/agent_graph.py` — LangGraph `StateGraph` definition; wires all nodes and edges
- `app/agent/__init__.py` — Exports the shared LLM instance (`ChatGoogleGenerativeAI`, `gemini-2.5-flash`)
- `app/agent/router.py` — Entry node; uses LLM to classify query → `vectorstore | web_search | llm_fallback`
- `app/agent/retrieval_grader.py` — Grades retrieved docs for relevance
- `app/agent/generate.py` — RAG generation chain; decides whether to retry or fall back
- `app/agent/hallucination_grader.py` — Checks generation for hallucinations and answer quality
- `app/agent/web_search.py` — Tavily search integration
- `app/agent/llm_fallback.py` — Direct LLM response (no retrieval)
- `app/vectorestore/__init__.py` — Pinecone vector store setup and similarity retrieval
- `app/variables.py` — Environment variable validation (fails fast on missing keys)
- `app/utils.py` — Converts `{"role": ..., "content": ...}` dicts to LangChain message objects

### GraphState

```python
{
  "messages": list,       # LangChain message objects (conversation history)
  "documents": list,      # Retrieved/searched docs
  "generation": str,      # Final LLM response
  "total_tokens": int,    # Accumulated token usage
}
```

### API

- `POST /chat` — `{"messages": [{"role": "user", "content": "..."}]}` → `{"message": "...", "usage": {"total_tokens": N}}`
- `GET /health` — `{"status": "ok"}`

## Environment Variables

Required in `.env`:

```
PINECONE_API_KEY=
GOOGLE_API_KEY=
TAVILY_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=
LANGCHAIN_PROJECT=
LANGCHAIN_API_KEY=
```

## Deployment

GitHub Actions (`.github/workflows/ci.yml`) builds the Docker image on every push and deploys to a VPS via SSH on pushes to `main`. Deploy uses `docker compose down && docker compose up -d --build`.
