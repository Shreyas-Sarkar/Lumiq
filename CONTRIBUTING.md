# Development workflow

Read `AGENTS.md` for boundaries and `CONTEXT.md` for navigation. Local tickets
follow `docs/agents/issue-tracker.md`.

## Setup

Use a Python version compatible with `backend/runtime.txt`. Backend imports
expect `backend/` as working directory.

```sh
cd backend
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env
uvicorn main:app --reload
```

Fill credentials before starting. When needed, provision the migration in your
development Supabase project after inspecting existing schema. Its `CREATE TABLE
IF NOT EXISTS` clauses do not upgrade existing table definitions.

In another terminal:

```sh
cd frontend
npm ci
cp -n .env.example .env.local
npm run dev
```

Fill frontend public Supabase configuration and API URL. Preserve existing
environment files. Provider and service-role keys stay on the backend and never
use `NEXT_PUBLIC_` names.

## Offline verification

From the repository root:

```sh
bash scripts/check.sh
```

This checks Python syntax without importing services or writing bytecode, then
TypeScript with incremental output disabled. Requires Python 3 and installed
frontend dependencies. No provider calls or package installs. Missing prerequisites
or check failures exit nonzero. It does not establish runtime correctness.

For frontend implementation changes, also run `npm run build` in `frontend/`;
it writes `.next/`. `npm run lint` currently has no checked-in ESLint configuration
and may prompt to initialize one; it is not the automated baseline.

## Behavioral verification

There is no checked-in application test suite yet. Add focused regression tests
for meaningful changes at public boundaries; use fake provider/database seams
offline. Prioritize routing, contextual follow-ups, charts, ownership, failed
execution, and quota interruption/recovery.

Live integration checks need a small bounded request budget. Report provider,
model, outcome, and limits without credentials. `/health` has a known status bug;
see `docs/known-issues.md` before treating it as deployment validation.

## Reviewing changes

Separate application changes from agent/documentation setup. Reconcile backend
and frontend contracts and update affected context documents. Preserve the MIT
license and provenance when updating vendored skills. Describe verification
evidence and remaining uncertainty in the handoff.
