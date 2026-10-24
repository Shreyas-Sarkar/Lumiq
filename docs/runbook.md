# Configuration and demo runbook

## Release order

1. Verify this branch's backend tests, production frontend build, browser journeys,
   and SQL checks. Keep Render/Vercel on their current release during preparation.
2. Inspect the target Supabase project and audit duplicate datasets:
   `select chat_id,count(*) from public.dataset group by chat_id having count(*)>1;`
   The migration intentionally stops if duplicates exist; never delete them blindly.
3. On an existing project, apply `supabase/migrations/20261002083254_investigation_workspace.sql`.
   A fresh project first needs `backend/db/migrations/001_initial_schema.sql`.
   Apply the whole migration in one transaction. It adds profiles, runs, budgets,
   leases, owner-read policies, and a private `lumiq-datasets` bucket. Dataset
   metadata becomes server-authored; browser inserts/updates/deletes are revoked.
4. Set backend configuration below, then deploy the backend release. `/health`
   must find the database, run table, and private bucket. Provider configuration
   status is not proof of live provider availability.
5. Deploy the frontend with Vercel's **Root Directory = frontend**, Node 22+,
   build `npm run build`. Use the Render HTTPS URL as `NEXT_PUBLIC_API_BASE_URL`.
   Root `vercel.json` is an older combined-service proposal, not this release's
   deployment configuration; the frontend root excludes it.
6. Run the live demo checklist before marking the release demo-ready.

Do not apply migrations or change production deployment merely because a branch
was pushed. The implementation specification reserves production release as an
explicit owner action. The SQL has been verified locally; a production rollout
must still check the actual target project's grants and storage behavior.

## Activation record — 2026-10-23

With explicit owner approval, the additive migration was applied in one transaction
on personal project `oanvayxyzjpwqsndctgg`, after checking for duplicate datasets.
The local backend enables investigations and disables legacy execution. Browser
checks with real providers verified private uploads, persisted runs, and recovery;
see `verification.md`. Render/Vercel have not been deployed by this work.

Local development needs both servers: run `uvicorn main:app --reload` from
`backend/` and `npm run dev` from `frontend/`. Starting the frontend alone does
not start the API on port 8000.

## Backend configuration

Render Root Directory: `backend`. Install `pip install -r requirements.txt`.
Start `uvicorn main:app --host 0.0.0.0 --port $PORT`. Python 3.11 is verified.

- `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`: backend-only database/storage access.
- `SUPABASE_ANON_KEY`: existing compatibility configuration; never substitute
  the service-role key in frontend configuration.
- `INVESTIGATIONS_ENABLED=true` only after the migration; `LEGACY_EXECUTION_ENABLED=false`.
- `DATASET_BUCKET=lumiq-datasets`; `MAX_FILE_SIZE_MB=5` (the route enforces a 5 MB ceiling).
- `ALLOWED_ORIGINS`: comma-separated exact frontend origins, e.g. localhost and
  the production Vercel custom domain. Add intentional preview origins explicitly.
- `GEMINI_API_KEY`, `GEMINI_MODEL=gemini-3.5-flash-lite`.
- `LLM_API_KEYS`: Groq key(s); only one organization budget is assumed.
  `LLM_MODEL=openai/gpt-oss-20b` was live-tested. The older configured
  `llama-3.3-70b-versatile` was no longer listed for this account.
- `GEMINI_RPM`, `GEMINI_RPD`, `GEMINI_TPM`, and corresponding `GROQ_*`:
  set from actual account limits. Defaults intentionally reserve little capacity.
  Reservations include estimated input and maximum output; outside usage is unknown.

Keys stay in ignored backend environment files or Render environment variables.
No paid fallbacks or hosted orchestration subscriptions are used.

## Provider failures and recovery

At most two HTTP attempts per stage, six per run, with one plan repair. An
attempt is checkpointed before network dispatch. Validated findings must reference
existing evidence; numerical facts come from evidence, not model prose.

- Quota: saves wait state, honors supported retry delay hints, then offers resume.
  Daily local reservations reset on Pacific calendar boundaries.
- Timeout/outage: tries the other configured provider within budget. If exhausted,
  saves an explicit reason. Completed evidence stays downloadable.
- Invalid output: validates the same contract and tries the other provider. If
  both fail, manual resume is available within the existing six-attempt budget.
- Authentication/model configuration: disables that provider in `provider_budget`.
  After fixing the key/model, explicitly clear its `disabled_reason` and expired
  `cooldown_until`. Never clear an active quota cooldown to bypass provider limits.
- Budget exhausted: start a new revision with the saved plan. A new revision is a
  distinct user action and uses the same provider-level quota reservations.

A backend restart or lost connection retains stage checkpoints. Leases expire
in 120 seconds; a poll or startup recovery can claim unfinished work. A lost
provider response consumes an attempt even if its result cannot be recovered.
There is no guaranteed completion while free Render sleeps.

## Merge and deployment preflight

Before merging an auto-deploying main branch, verify the target hosting settings:
Render root `backend`, workspace flag enabled, legacy execution disabled, current
provider configuration, private dataset bucket, and exact allowed frontend origins.
Vercel root must be `frontend`, with its public API URL pointing to the Render
HTTPS origin. The root combined-service `vercel.json` is historical.

The hosted Supabase migration was applied with approval on 2026-10-23. That does
not establish Render/Vercel environment readiness. A green Vercel build does not
prove upload, inference, persistence, or cold-start recovery against the deployed API.
Keep a production deployment smoke pass distinct from code merge review.

## Storage and retention

CSV paths are generated as `owner-id/dataset-id.csv`. The bucket is private and
backend-only; the API checks ownership and validates the object path. Old local
files are read only inside the authenticated owner's upload directory. Missing
historical files require a new upload; metadata cannot recreate lost bytes.

Tool version 1.1 preserves unknown grouped sums and contributions. New runs cannot
reuse older calculation keys; unfinished older evidence is recomputed before the
analyst. Completed historical runs stay unchanged; revise their plan for updated
calculations. Exact validated interpretation results can be reused from completed
owner/dataset-scoped state; different context or configured models require fresh
interpretation. Reuse is disclosed in run history and consumes no inference attempt.

Run status history is capped at thirty events; each state stores only its latest
checkpoint. Evidence is bounded to four steps and fifty displayed rows per step.
Saved runs and objects are retained until deliberately removed. Monitor the free
Supabase storage allowance. Before cleanup, identify unreferenced objects and
abandoned failed/cancelled runs; review the list, then remove only those records.
Automatic destructive retention is intentionally not enabled.

## Live demo checklist

Use `samples/revenue.csv`, which contains synthetic data and known answers.

- Sign in with a test account and create a workspace.
- Upload the CSV, confirm private durable bytes and profile, then restart the
  backend and reload it. A local disk-only check is insufficient.
- Ask an ambiguous question, answer the clarification, then compare January and
  February 2026 revenue by region. Expect 300 → 340, +40; North +60, South −20.
- Inspect evidence, limits, table/chart together, CSV download, saved-run reload,
  and an edited date/filter revision.
- Run one bounded request with each configured provider. Test failover both ways
  using controlled errors; never intentionally exhaust real free quota.
- Verify all-provider failure preserves evidence and shows a truthful reason.
- Verify expired/renewed browser sessions and a backend cold start. Wait for the
  free service to wake naturally; do not add a keep-alive quota workaround.
- Keep a clearly labelled saved result available as the demo fallback. Saved
  evidence is not presented as a fresh live inference run.

If a required check fails, keep the working previous deployment, retain this
branch for review, and report the failing dependency/configuration. Do not
silently enable the legacy arbitrary-code endpoint as a fallback.
