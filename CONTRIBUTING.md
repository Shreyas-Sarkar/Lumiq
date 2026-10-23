# Development

Read `AGENTS.md` and `CONTEXT.md`. Use Python 3.11+ and Node.js 22+.

```sh
cd backend
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp -n .env.example .env
```

Populate backend credentials without overwriting an existing environment. Follow
`docs/runbook.md` for the database migration/private bucket and workspace flag.
Start `uvicorn main:app --reload` from `backend/`.

```sh
cd frontend
npm ci
cp -n .env.example .env.local
npm run dev
```

Set the public Supabase URL/anon key and backend API URL. Secrets never belong in
`NEXT_PUBLIC_` variables. Frontend builds use webpack because Turbopack workers
could not bind their local evaluation port in the development sandbox.

## Verification

From the root, `bash scripts/check.sh` checks Python syntax and frontend types.
`bash scripts/test.sh` runs the offline Python behavior suite. It uses
`backend/.venv` or the existing `backend/venv`; set `LUMIQ_TEST_PYTHON` to override.

Run `npm run build` from `frontend/`. With the frontend running on localhost:3000,
run `npm test` there for browser journeys. The suite mocks auth/API boundaries and
never uses live model quota. Screenshots/traces go to ignored `output/playwright/`.

`python scripts/benchmark.py` uses known-answer sample data and measures local
calculation time and peak process memory. `scripts/check-database.sql` runs against
a disposable Postgres database after the migrations; it rolls back its fixtures.
It expects Supabase-compatible auth/storage schemas (or a local test bootstrap).

From `backend/`, `python ../scripts/smoke-models.py` makes up to two small requests
per configured provider. Run only as an intentional live check; no key is printed.
Record the provider/model/outcome and distinguish access checks from analytic evals.

## Changes

Use descriptive branch names without an assistant-name prefix. GitHub writes use
only `Shreyas-Sarkar`; verify login and remote before each push. Make small verified
commits. Reconcile frontend/backend contracts, update affected context, and retain
skill provenance/licenses. Deployment requires the concrete release checks in the
runbook; a green build alone does not establish live behavior.
