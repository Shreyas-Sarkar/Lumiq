# Lumiq — AI-Powered Conversational Data Analysis Platform

## Product Name
**Lumiq** (Luminous + IQ) — illuminate the intelligence inside your data.

---

## 1. Problem Definition

### The Core Problem
Data analysts, product managers, researchers, and domain experts spend enormous time translating
business questions into code. To ask "Which region had the highest returns last quarter?" from a
CSV file, a non-technical user must know Pandas, Python, and how their data is structured.
Alternatively, they must wait for an analyst to write code for them.

Existing solutions either:
- Require technical expertise (Jupyter notebooks, SQL consoles)
- Hallucinate results without grounding (raw GPT-4 over CSV)
- Are expensive, closed, or require cloud data uploads to proprietary systems
- Provide no transparency into how results were derived
- Do not preserve conversation context for iterative querying

### Real-World Pain Points
1. A product manager cannot self-serve exploratory analysis on their metrics CSVs.
2. A researcher wants to ask 10 follow-up questions about a dataset without re-uploading or rewriting queries.
3. A business analyst needs reproducible, code-backed answers — not AI-guessed summaries.
4. Non-engineers cannot evaluate whether an AI answer is correct because they can't see the code.

---

## 2. Target Users

| User Type | Description |
|---|---|
| Product Managers | Analyze product metrics, funnels, feature adoption without writing code |
| Business Analysts | Iterate on data exploration with full code transparency |
| Data Scientists | Rapidly prototype queries and audit AI-generated code |
| Researchers | Query academic or experimental datasets conversationally |
| Ops / Finance Teams | Slice and analyze financial or operational CSV exports |

**Non-Target Users:** Enterprise data teams requiring database connections, real-time streaming
data, or multi-source joins. This MVP is strictly CSV-based.

---

## 3. Why Existing Solutions Fail

| Solution | Failure Reason |
|---|---|
| ChatGPT + CSV | Hallucinates statistics; no code execution; no persistent sessions |
| Jupyter Notebooks | Requires Python knowledge; no conversation interface |
| Google Sheets / Excel AI | Limited analysis depth; no complex aggregations; no transparency |
| Pandas AI (open source) | No UI; no auth; no persistence; developer-only tool |
| Tableau / Power BI | Expensive; requires data modeling expertise; not conversational |
| Julius.ai | Paid/closed; no self-hosted option; no RAG grounding |

**The Gap:** There is no free-tier, open-stack, code-transparent, RAG-grounded, conversational
data analysis tool with persistent multi-session chat. Lumiq fills this gap.

---

## 4. Proposed System

Lumiq is a full-stack web application where users:
1. Authenticate (Supabase Auth)
2. Upload a structured CSV dataset per chat session
3. Ask natural language questions about their data
4. Receive answers grounded in executed Pandas code, with full transparency

### How It Works (High Level)
```
User Query
    → RAG retrieves schema/sample context from ChromaDB
    → LLM generates Pandas code using retrieved context + chat history
    → Execution Engine runs code in a controlled Python scope
    → Evaluator validates result relevance and correctness
    → Explanation Layer formats: Answer + Code + Reasoning + Assumptions
    → Response streamed back to ChatGPT-style UI
```

### Core Guarantee
**Every answer is derived from executed code. Zero hallucinated statistics.**

---

## 5. Key Features

### Authentication & Multi-User
- Supabase Auth (email/password or magic link)
- All data strictly scoped per user (no cross-user leakage)

### ChatGPT-Style Conversational UI
- Left sidebar: list of all chats with titles and timestamps
- Main panel: conversation thread with messages
- Input box at bottom with animated processing phases
- Code blocks with syntax highlighting
- Table rendering for DataFrame outputs
- Animated status indicators (6 processing phases)

### Dataset Upload & Indexing
- Upload CSV per chat session
- Auto-extract: column names, dtypes, sample rows, summary statistics
- Embed schema into ChromaDB vector store
- Preview dataset metadata in the chat

### RAG-Grounded Code Generation
- ChromaDB stores embeddings of: schema, samples, column descriptions, summary stats
- Every query retrieves the most relevant schema context before LLM call
- LLM receives: User Query + Retrieved Schema Context + Chat History
- Prevents hallucination of non-existent column names or wrong data types

### Safe Code Execution
- Generated Pandas code executed in a controlled Python scope
- Allowed imports: `pandas`, `numpy`, `datetime`, `math`, `statistics`, `re`
- Blocked: file I/O, subprocess, os, sys, network calls, `__import__`
- DataFrame is pre-loaded into execution scope as `df`
- Execution timeout enforced (10 seconds)

### Evaluation & Retry
- LLM evaluator checks: result non-null, result relevant, no error output
- Retry up to 2 times on failure with error feedback injected into next prompt
- On persistent failure: graceful rejection with explanation

### Transparent Response Format
Every response includes:
- **Answer**: Human-readable summary of the result
- **Code**: The exact Pandas code executed
- **Result**: The actual output (table, scalar, list)
- **Reasoning**: Step-by-step explanation of approach
- **Assumptions**: Any assumptions made about the data

### Persistent Chat Sessions
- Unlimited chats per user
- Each chat has: its own uploaded dataset, its own message history
- Resume any past chat and continue querying the same dataset
- Chat titles auto-generated from first query

### Logging
- Every execution logged: query, generated code, output, timestamps, errors
- Stored in Supabase `execution_logs` table

---

## 6. Constraints

### Hard Technical Constraints
- **Free-tier only**: Supabase free tier, HuggingFace embeddings (no OpenAI embeddings), ChromaDB in-process, any free LLM API (Groq/Gemini free tier / local Ollama)
- **No unsafe code execution**: No Docker sandbox; controlled `exec()` with restricted globals
- **No arbitrary imports**: Only whitelisted Python stdlib + Pandas + NumPy
- **CSV only**: No Excel, JSON, Parquet, or database connections in MVP
- **Single file per chat**: One dataset per chat session; no multi-file joins

### Scope Boundaries (Non-Goals)
- No real-time or streaming datasets
- No multi-table joins across chats
- No PDF/image/document analysis
- No voice input
- No export to PDF/Excel
- No embedding fine-tuning
- No multi-user collaboration on same dataset
- No enterprise SSO
- No full Docker sandbox / gVisor
- No deployment automation (CI/CD pipelines out of scope)
- No mobile-native app (responsive web only)

---

## 7. Tech Stack (Canonical, Do Not Deviate)

| Layer | Technology |
|---|---|
| Frontend | Next.js 14 (App Router), TypeScript, Tailwind CSS |
| Backend | Python 3.11+, FastAPI, Uvicorn |
| Database | Supabase (PostgreSQL) — users, chats, messages, execution_logs |
| Auth | Supabase Auth (email/password) |
| Vector DB | ChromaDB (in-process, persistent local storage) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` via `chromadb` |
| LLM | Groq API (llama3-8b-instruct, free tier) OR Google Gemini free tier |
| Code Execution | Python `exec()` with restricted `__builtins__` dict |
| Dataset Processing | Pandas, NumPy |
| Syntax Highlighting | `react-syntax-highlighter` (frontend) |
| HTTP Client | `axios` (frontend), `httpx` (backend) |

---

## 8. Project Structure (Reference)

```
lumiq/
├── backend/
│   ├── main.py                  # FastAPI app entry
│   ├── routers/
│   │   ├── upload.py            # POST /upload
│   │   ├── chat.py              # POST /chat, GET /chats, GET /chat/{id}
│   │   └── auth.py              # Auth middleware helpers
│   ├── services/
│   │   ├── dataset_manager.py   # CSV loading, schema extraction
│   │   ├── chat_manager.py      # Chat CRUD, message persistence
│   │   ├── code_generator.py    # LLM prompt + code generation
│   │   ├── execution_engine.py  # Controlled exec() runtime
│   │   ├── evaluator.py         # Output validation + retry logic
│   │   ├── rag_service.py       # ChromaDB ops, embedding, retrieval
│   │   └── explanation.py       # Response formatting
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   ├── db/
│   │   └── supabase_client.py   # Supabase Python client init
│   └── config.py                # Env vars, settings
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx             # Root redirect
│   │   ├── login/page.tsx
│   │   ├── signup/page.tsx
│   │   └── chat/
│   │       ├── page.tsx         # Chat list / new chat
│   │       └── [id]/page.tsx    # Individual chat view
│   ├── components/
│   │   ├── Sidebar.tsx          # Chat list sidebar
│   │   ├── ChatPanel.tsx        # Main chat message area
│   │   ├── MessageBubble.tsx    # Individual message renderer
│   │   ├── InputBar.tsx         # Query input + upload
│   │   ├── CodeBlock.tsx        # Syntax-highlighted code
│   │   ├── DataTable.tsx        # DataFrame table renderer
│   │   ├── ProcessingPhases.tsx # Animated status indicator
│   │   └── UploadModal.tsx      # CSV upload UI
│   ├── lib/
│   │   ├── supabase.ts          # Supabase client
│   │   └── api.ts               # Backend API calls
│   └── types/
│       └── index.ts             # Shared TypeScript types
├── .env.example
├── requirements.txt
├── package.json
└── README.md
```
