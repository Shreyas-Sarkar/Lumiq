# Working on Lumiq

Lumiq is a CSV analysis application. The current investigation path uses a two-specialist LangGraph and trusted
analytical tools. Historical generated-Python execution is disabled by default.

## Start here

- Read `CONTEXT.md` when tracing a flow or changing architecture.
- Read `GLOSSARY.md` when naming domain concepts. Read relevant `docs/adr/`
  records before changing their area.
- Read `docs/known-issues.md` when fixing routing, caching, execution, charts,
  authentication, persistence, or health reporting.
- Read `CONTRIBUTING.md` for setup and verification. Run `bash scripts/check.sh`
  for the offline baseline; it checks syntax and types, not runtime correctness.
- Read `.scratch/investigation-workspace/spec.md` for the original redesign scope.
  A spec is not evidence of implemented behavior or authorization to deploy.

## Implementation boundaries

- Investigation payloads live in `backend/models/investigations.py` and
  `frontend/types/investigations.ts`; API calls live in `frontend/lib/api.ts`.
  Reconcile both contracts when changing payloads. Legacy schemas remain separate.
- Backend database calls use a service-role client. Preserve explicit user/chat
  ownership checks at API boundaries, even with owner RLS in the migration.
- Preserve hard failure after exhausted computation retries: return an error
  rather than asking the model to invent a data-derived answer.
- Successful execution is not proof of statistical correctness. Distinguish
  computed evidence, interpretation, and limitations.
- Legacy execution is regex-filtered, in-process `exec()` with a thread timeout.
  Treat it as a restricted prototype, not an isolated sandbox.
- Qdrant/Cohere retrieval is optional. `chroma_collection_id` is a legacy name
  used by that implementation, not evidence that ChromaDB is installed.

## Resources and credentials

- Constraints: Render free backend, Vercel frontend, free rate-limited APIs.
  Reduce model calls/tokens; bound retries, dataset memory, and concurrency.
- Use ignored backend environment files or backend deployment variables for
  secrets. Report presence only; never print values or authenticated headers.
- Gemini and Groq have live-tested adapters. Environment entries alone do not
  prove current availability; keep integration checks bounded. Jev is excluded by the owner.
- Mock external providers in automated tests. Keep requested live integration
  checks small and report their request budget and validation limits.
- Render disk and process memory are temporary. Durable datasets, job state,
  checkpoints, and cooldowns must survive restart and sleep elsewhere.

## Change discipline

Use only the `Shreyas-Sarkar` GitHub account for writes to this repository.
Verify the authenticated login and repository remote before each push or other
GitHub mutation; switch to the owner's account if needed. Use plain descriptive branch names without an assistant-name prefix. Do not
mention the coding assistant in branch names, application code, or project-authored
documentation. This naming rule overrides tool defaults. Keep commits small and
coherent. Run checks appropriate to each change and push only verified work;
report validation limits rather than claiming a green typecheck proves live behavior.

Use existing module boundaries before adding abstractions. Verify observable
failures and add focused regression tests for meaningful behavior changes.
Documentation-only work needs document/link checks, not new behavior tests.
Report commands run and what they establish; update context when behavior changes.

## Provider resilience and demos

Gemini and Groq are both approved providers for the investigation runtime. Use a common
validated contract, provider-specific quota state, and bounded failover. Preserve
completed evidence across inference failures. If every eligible provider fails,
communicate the actual failure category and whether work can resume. Demo readiness
requires real end-to-end flows and recovery checks, not only mocked inference.

## Agent skills

### Issue tracker

Specs and tickets are local Markdown under `.scratch/<feature>/`. Read
`docs/agents/issue-tracker.md` when creating, fetching, or updating a ticket.

### Domain docs

Use one root `GLOSSARY.md` and `docs/adr/`. Read `docs/agents/domain.md` when
creating or updating domain documentation.

### Installed workflows

Project-local Matt Pocock skills live in `.agents/skills/`; provenance and
selection are in `.agents/skills/SOURCES.md`. Use the relevant skill when its
scope fits. Setup and spec synthesis are explicitly invoked workflows.
Skill installation is not authorization to publish issues, send messages,
change deployment settings, or spend unbounded provider quota.
