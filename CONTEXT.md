# Lumiq: current implementation

Verified against source on 2026-10-02. Update this map when behavior changes.

## Product

A signed-in user creates a chat, uploads one UTF-8 CSV, and asks questions.
Responses expose computed results, generated code, model interpretation, and
follow-up questions. Database connectors, multi-dataset joins, durable background
jobs, a multi-agent graph, and Gemini inference are not implemented.

## Repository map

| Area | Entry point | Responsibility |
| --- | --- | --- |
| Frontend routes | `frontend/app/` | Landing, signup/login, chat list/detail |
| Active analysis UI | `frontend/components/ChatPanel.tsx`, `AnalysisBlock.tsx` | Messages, results, charts, follow-ups |
| API/auth clients | `frontend/lib/` | Axios calls with bearer tokens; browser Supabase client |
| API startup | `backend/main.py`, `config.py` | FastAPI lifespan and service wiring |
| API routes | `backend/routers/` | Auth, ownership, queries, CSV upload |
| Query pipeline | `backend/services/orchestrator.py` | Routing, cache, retries, persistence |
| Computation | `code_generator.py`, `execution_engine.py`, `evaluator.py` | Python synthesis, execution, relevance check |
| Interpretation | `cognitive_engine.py`, `explanation.py`, `context_summarizer.py` | Insights, formatting, recent-query summary |
| Dataset context | `dataset_manager.py`, `rag_service.py`, `embedding_service.py`, `vector_store.py` | CSV profile, Cohere embeddings, Qdrant retrieval |
| Persistence | `backend/db/`, `chat_manager.py`, `user_sync.py`, `log_service.py` | Schema, user mirror, chats/messages/logs |
| Request control | `llm_client.py`, `query_cache.py` | Serialized queue and process-local cache |

Unqualified backend service names are relative to `backend/services/`.
`MessageBubble.tsx` and `ProcessingPhases.tsx` appear unused in the active
frontend flow; confirm callers before removing them.

## Upload

`POST /api/upload` authenticates the user, checks chat ownership and existing
dataset presence, reads/validates the CSV, writes it locally, and saves schema,
samples, statistics, and storage path in Supabase. Optional indexing embeds three
documents (schema, samples, stats) into Qdrant. Its failures do not fail upload.

Default ceiling: 50 MB; validation rejects more than 500 columns. These are not
measured memory/concurrency budgets. The database does not enforce one dataset
per chat with a uniqueness constraint; concurrent uploads can pass the guard.

## Query

1. `POST /api/chat` validates token, ownership, query format, and length.
2. The orchestrator saves the user message, may update the title, and loads CSV.
3. Cache lookup uses normalized query and a path/size/mtime fingerprint.
4. A heuristic chooses `executor`, `hybrid`, `concept`, or `irrelevant`.
5. Recent messages feed a summary. Optional RAG retrieval runs for non-executor
   modes and is skipped for executor mode.
6. Executor generates, safety-checks, executes, and evaluates code, with retries.
   Hybrid generates and executes but skips the separate relevance check.
   Concept explains without query-specific execution; irrelevant rejects.
7. Exhausted computation failures hard-fail. Successful execution feeds model
   interpretation and response formatting.
8. Responses are cached and saved as assistant metadata. Logs are best-effort;
   cache hits skip normal execution logging.

Tables serialize at most 50 display rows. Cognitive input is truncated further.
Charts are base64 PNGs; a chart accompanying a table/scalar result is currently
omitted by the API formatter. Progress labels are timer-driven, not server events.

## Infrastructure

- Next.js/React; FastAPI/Pandas/Matplotlib.
- Supabase Auth/Postgres. Backend uses service-role, browser uses anon. Explicit
  application ownership checks remain necessary despite migration RLS.
- CSV bytes live on local disk; records in Supabase; vector context in Qdrant.
- Model requests use an OpenAI-compatible chat-completions endpoint. Startup
  requires comma-separated `LLM_API_KEYS`. Queue and cache are process-local.
- Cohere `embed-english-v3.0` supplies embeddings; vector services initialize lazily.
- Gemini credentials are provisioned locally, but no Gemini adapter is implemented.
  Jev is excluded from planned work by the owner.

## Authority

Source/configuration establish current behavior. This is the navigation map;
`docs/known-issues.md` records limitations; the local investigation spec records
future work. `idea.md`, root diagram files, and `vercel.json` are historical
design/deployment material, not verified production state. Inspect actual
deployment and database state before making claims about them.
