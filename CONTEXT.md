# Lumiq implementation map

Backend verified against source on 2026-10-02; frontend updated on 2026-10-23.

## Flow

1. Supabase authenticates the browser; the backend validates bearer tokens and
   verifies workspace ownership on each public operation.
2. Upload validates a bounded CSV, stores its bytes in a private owner-scoped
   Supabase object, and records an immutable content hash and deterministic profile.
3. Creating an investigation persists a queued run and returns its ID.
4. LangGraph runs planner → trusted tools → analyst. A schema-validated plan may
   instead ask for clarification or return a conceptual explanation.
5. Explicit analytical tools validate columns/types/filters/periods and compute
   evidence. One plan repair is allowed; failed calculations produce no findings.
6. The analyst receives evidence and must return valid evidence references without
   numerical claims. Numeric facts in the UI come from the stored calculations.
   Period comparison direction and row-count statements are grounded deterministically
   on creation and when reading historical runs.
7. Gemini and Groq share a bounded attempt budget. Provider reservations, cooldowns,
   run state, and expiring leases persist in Postgres through owner-scoped RPCs.
8. The browser polls stored status. Resume preserves completed evidence. Startup
   recovers up to three unfinished runs; sleeping hosting has no background guarantee.

## Files

| Area | Entry points |
| --- | --- |
| Homepage | `frontend/app/page.tsx`, `home.css`, `frontend/components/SampleInvestigation.tsx` |
| Workspace UI | `frontend/components/InvestigationWorkspace.tsx`, `PlanEditor.tsx` |
| Browser session | `frontend/hooks/useSession.ts`, `frontend/lib/supabase.ts` |
| API / UI contracts | `frontend/lib/api.ts`, `frontend/types/investigations.ts` |
| Backend startup / flags | `backend/main.py`, `backend/config.py` |
| Authenticated routes | `backend/routers/upload.py`, `investigations.py`, `chat.py` |
| Plan validation | `backend/models/investigations.py` |
| Profiling / calculations | `backend/services/analysis_tools.py` |
| Two-specialist graph | `backend/services/investigation.py` |
| Provider adapters / budgets | `backend/services/model_gateway.py` |
| Durable state / leases | `backend/services/run_store.py`, `supabase/migrations/` |
| Durable CSV bytes | `backend/services/dataset_manager.py` |
| Offline API tests | `backend/tests/` |
| Browser journeys | `frontend/tests/workspace.spec.ts`, `home.spec.ts` |

The graph uses application stage checkpoints, not LangGraph PostgresSaver or
checkpoint replay; see ADR 0002. Persisted state holds references and bounded
evidence, not dataframes. Completed runs are immutable through the public API;
revisions share an investigation ID. Computations can be reused for identical
owner, dataset version, plan, and tool version. Interpretation is not reused across
changed context.

## Legacy and deployment

Historical `ChatPanel`, code generation/execution, RAG, and confidence metadata
remain for compatibility. New workspace routes use `InvestigationWorkspace`;
`AnalysisBlock` reads old messages. Legacy execution requires the explicit
`LEGACY_EXECUTION_ENABLED` flag and remains unsafe for arbitrary public code.
Optional Qdrant/Cohere is unnecessary on the investigation path.

`idea.md` and root diagrams describe earlier designs. Deployment state is separate
from source; `docs/runbook.md` records setup, migration, and live verification.
