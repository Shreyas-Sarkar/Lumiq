# Lumiq implementation map

Backend and frontend updated on 2026-10-24; see docs/verification.md for test scope.

## Flow

1. Supabase authenticates the browser; the backend validates bearer tokens and
   verifies workspace ownership on each public operation. Workspace names can be
   changed through the owner-filtered rename endpoint; trimmed names are 1–120 characters.
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
   on creation and when reading historical completed runs. Failed interpretation
   keeps evidence available without fabricating completed findings.
   All-missing grouped sums remain unknown; only categories absent from a period
   contribute zero. Grouped evidence containing unknown measurements gets a
   deterministic limitation statement so model prose cannot rank it as zero.
   Tool version 1.1 invalidates old computation reuse; unfinished
   older evidence is recomputed before interpretation.
7. Gemini and Groq share a bounded attempt budget. Provider reservations, cooldowns,
   run state, and expiring leases persist in Postgres through owner-scoped RPCs.
8. The browser polls stored status. Resume preserves completed evidence. Startup
   recovers up to three unfinished runs; sleeping hosting has no background guarantee.
   Resume/cancel revalidate the freshly claimed state, preserving newer completed
   evidence and attempt counts. Changed clarification invalidates dependent results.

## Files

| Area | Entry points |
| --- | --- |
| Homepage | `frontend/app/page.tsx`, `home.css`, `frontend/components/SampleInvestigation.tsx`, `InvestigationTrace.tsx`, `HomeMotion.tsx` |
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
owner, dataset version, plan, and tool version. Validated interpretation payloads
are cached separately in completed run state, scoped to owner/dataset and keyed by
the exact question/context/evidence prompt, schema, prompt version, evidence/tool
key, and configured provider models. Cache hits are validated again and disclosed
in run history; internal payloads/keys are not returned by the public API. Changed
context or model configuration requires fresh interpretation. Planner calls are
not cached.

## Legacy and deployment

Historical `ChatPanel`, code generation/execution, RAG, and confidence metadata
remain for compatibility. New workspace routes use `InvestigationWorkspace`;
`AnalysisBlock` reads old messages. Legacy execution requires the explicit
`LEGACY_EXECUTION_ENABLED` flag and remains unsafe for arbitrary public code.
Optional Qdrant/Cohere is unnecessary on the investigation path.

`idea.md` and root diagrams describe earlier designs. Deployment state is separate
from source; `docs/runbook.md` records setup, migration, and live verification.
