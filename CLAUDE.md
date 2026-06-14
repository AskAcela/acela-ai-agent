# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
uv sync                                              # install dependencies from uv.lock
uv run uvicorn app.main:server --reload              # run the API server locally (port 8000)
uv run python index_documents.py                     # (re)build the Pinecone vector index from source docs
docker compose up -d --build                         # run the server in Docker (reads .env)
```

There is no test suite or linter configured. CI (`.github/workflows/ci.yml`) only builds the Docker image and, on push to `main`/`master`, SSH-deploys to a VPS via `git pull` + `docker compose up -d --build`.

A `.env` file is required. `validate_environment()` (`app/variables.py`) hard-fails at startup if any of these are missing: `LANGCHAIN_TRACING_V2`, `LANGCHAIN_ENDPOINT`, `LANGCHAIN_PROJECT`, `LANGCHAIN_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`, `PINECONE_API_KEY`. The submitter agent additionally needs `UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN` (read via `Redis.from_env()`).

## Architecture

A FastAPI server (`app/main.py`) exposing two **separate and unrelated** AI systems. Understanding the codebase means understanding that `/chat` and `/chat/submit` share almost nothing.

### 1. CRAG question-answering graph — `POST /chat`

A Corrective-RAG pipeline built as a LangGraph `StateGraph` in `app/agent_graph.py`, with each node living in its own file under `app/agent/`. Shared `GraphState` (a `TypedDict`) carries `messages`, `documents`, `generation`, and `total_tokens` between nodes. The graph is compiled **once at startup** (`createAgentGraph()` in `main.py`) and reused per request.

Control flow:
- **Router** (`router.py`) classifies the question into `vectorstore`, `web_search`, or `llm_fallback`.
- **vectorstore path**: `retrieve` (Pinecone) → `grade_documents` (LLM scores each doc yes/no, drops irrelevant ones) → if any survive `generate`, else fall through to `web_search`.
- **generate** (`generate.py`) runs RAG, then `grade_generation_v_documents_and_question` (`hallucination_grader.py`) checks the answer twice: grounded-in-docs? and answers-the-question? Verdicts route back to `generate` ("not supported"), to `web_search` ("not useful"), or to `END` ("useful").
- **llm_fallback** (`llm_fallback.py`) answers from model knowledge alone, straight to `END`.

Nodes return either a plain state dict or a `langgraph.types.Command(goto=..., update=...)` for conditional routing. Every LLM call accumulates `usage_metadata["total_tokens"]` into `total_tokens`, which is returned to the client.

The shared LLM for all graph nodes is defined in `app/agent/__init__.py` as `llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")`. Structured graders use `llm.with_structured_output(<PydanticModel>)`.

The vector store (`app/vectorestore/__init__.py` — note the spelling) is a lazily-initialized global Pinecone index (`acela-test-index`, 512-dim cosine) using `gemini-embedding-2-preview` embeddings. `index()` loads Celo docs from hardcoded GitHub URLs; `retrieve()` is the graph node.

### 2. Submitter agent — `POST /chat/submit`

An independent LangChain tool-calling agent (`app/submitter/`) that guides Celo Builders hackathon submissions against the `celobuilders.xyz` API. Entry point is `run_agent(session_id, chat_history)` in `agent.py`, called fresh per request (uses `InMemorySaver`, so no cross-request memory — durable state lives in Redis instead).

- **Session binding is the key pattern** (`_bind_session` in `agent.py`): the two tools are wrapped in `StructuredTool`s that close over `session_id`, so the model never sees or passes the session ID — it is injected server-side.
- **Tools** (`tools.py`): `http_request` (calls the Celo API, auto-injecting a bearer token read from the KV store) and `kv_store` (`set`/`get`/`list`/`delete` against an Upstash Redis store keyed by `session_id`, with the agent-chosen key nested inside a per-session dict).
- **Behavior** is driven entirely by the large system prompt / "skill document" in `prompt.py`, which enumerates the API endpoints, auth flow, and required submission fields. Changes to agent behavior usually mean editing `prompt.py`, not code.
- Uses its own LLM instance: `ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)`.

### Shared pieces

- `app/utils.py` — `convert_messages()` maps the incoming `{role, content}` JSON into LangChain message objects; used by both endpoints.
- `app/logger.py` — the `acela` logger used everywhere. Controlled by `LOG_LEVEL` and optional `LOG_FILE` env vars. Code logs heavily at INFO with `"Node: ..."` / `"Decision Node: ..."` markers tracing graph traversal.

## Gotchas

- `pyproject.toml` requires Python `>=3.14`, but the `Dockerfile` builds on `python:3.11-slim`. Local and container Python versions differ — keep this in mind when a dependency behaves differently in Docker.
- The vector store package is misspelled `vectorestore` — match it exactly in imports.
