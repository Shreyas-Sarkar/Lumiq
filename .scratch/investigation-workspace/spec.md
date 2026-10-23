# Lumiq investigation workspace

Status: implementation verified locally; live release pending
Label: ready-for-agent
Date: 2026-10-02
Planning workflow: Matt Pocock `to-spec`, pinned in this repository.

This is a reviewable implementation plan. Repository setup is complete; the
core application changes are implemented on the investigation-workspace branch.
See docs/verification.md for the verified scope and remaining live release gates.
The owner confirmed the
API behavior test seam plus a small set of browser journeys.

## Problem Statement

Lumiq can calculate from a CSV, but users cannot reliably inspect an analytical
plan, distinguish supported findings from interpretation, or recover work after
rate limits and backend restarts. The chat interface hides analytical state.
The current routing, contextual cache, chart handling, and evaluation introduce
avoidable errors. Local CSV storage is incompatible with durable free Render
hosting. Adding unconstrained model agents would increase latency and quota use.

## Solution

Create a saved investigation workspace with a dataset profile, editable analysis
plan, evidence-linked findings, interactive charts, and resumable execution.
Use two bounded model specialists: a planner selects validated analytical tools;
an analyst interprets their evidence. A deterministic graph controls routing,
validation, quota limits, and recovery. Retain the existing frontend/backend
foundation and use Gemini 3.5 Flash-Lite plus compatible available Groq models
interchangeably through a validated provider boundary with bounded failover.
Jev is excluded. Avoid mandatory embeddings and recurring LLM summaries.

The first complete investigation is period-over-period change analysis: identify
the metric and time window, compare periods, and compute group contributions.
Broader descriptive tools can reuse the same plan/evidence structure.

## User Stories

1. As a user, I want my existing account and chats to remain usable during the
   transition, so I retain previous work.
2. As a user, I want to upload a CSV with clear size and parsing errors, so I can
   correct a rejected file.
3. As a user, I want uploaded data to survive backend restarts, so I need not
   re-upload after every cold start.
4. As a user, I want a profile showing columns, types, missing values, duplicates,
   and date coverage, so I can understand data quality before asking questions.
5. As a user, I want to ask an analytical question in natural language, so I need
   not write code for common analysis.
6. As a user, I want Lumiq to ask which date, metric, or period I mean when
   ambiguous, so it does not silently choose a misleading interpretation.
7. As a user, I want to inspect the planned calculations and assumptions, so I
   know how my question will be answered.
8. As a user, I want to edit filters and time windows, so I can rerun an analysis
   without starting over.
9. As a user, I want to compare periods and segment contributions, so I can
   identify the measured sources of a change.
10. As a user, I want each finding to link to its evidence, so I can verify the
    underlying calculation.
11. As a user, I want observations separated from explanations of business causes,
    so I understand what my data can establish.
12. As a user, I want charts and tables to appear together, so I can inspect both
    the visual pattern and exact measurements.
13. As a user, I want warnings about incomplete periods, missing values, and small
    samples, so I can judge an answer's limitations.
14. As a user, I want conceptual questions answered without unnecessary
    calculation, so I can learn while conserving quota.
15. As a user, I want contextual follow-ups to preserve selected filters and
    metrics, so “what about that segment?” uses the intended context.
16. As a user, I want completed work reused when inputs are unchanged, so repeated
    questions do not consume unnecessary model calls.
17. As a user, I want real progress and quota-wait states, so I know whether work
    is queued, running, paused, or needs my input.
18. As a user, I want to resume after a network interruption or backend restart,
    so completed calculations are retained.
19. As a user, I want computation results to remain accessible if model
    interpretation is unavailable, so useful completed work is not discarded.
20. As a user, I want saved investigations and result-table downloads, so I can
    revisit and use the work outside the conversation.
21. As a user, I want my datasets and investigations isolated from other users,
    so private data is protected.
22. As a user, I want a usable narrow-screen experience with keyboard-accessible
    controls, so the workspace is not restricted to a large desktop.
23. As the owner, I want bounded calls, tokens, memory, and retries, so the app
    stays useful within free-tier limits.
24. As the owner, I want automated known-answer and recovery tests without live
    inference, so regressions are cheap to detect.

## Implementation Decisions

### Scope and stack

- Keep Next.js/React/Tailwind and FastAPI/Pandas with Supabase Auth/Postgres.
- Use self-hosted Python LangGraph for explicit execution branches and persistent
  checkpoints; no hosted agent platform subscription is required by the plan.
- Use Gemini `gemini-3.5-flash-lite` through a small provider boundary. Prefer the
  native API/SDK for structured outputs and quota errors, verifying exact
  capabilities before implementation. Do not layer a second agent runner inside
  the graph. Pin tested dependencies and log the model/prompt version.
- Retain Groq as an approved live provider, not only a migration path. Choose
  an eligible model per specialist based on tested capabilities, availability,
  and remaining quota. Both providers return the same validated plan/finding
  contracts. Record the provider/model used at each stage.
- Fail over on quota exhaustion, temporary provider outages/timeouts, or invalid
  structured output within a shared attempt budget. Keep cooldowns provider/model
  scoped so a Gemini limit does not block healthy Groq capacity. Disable a provider
  with rejected credentials and expose an owner-facing configuration reason.
  Input, authorization, and analytical errors are not fixed by indiscriminate
  provider switching. No paid fallback is approved.
- Reuse a successful plan or computed evidence when switching providers; change
  only the failed inference stage. Different providers must pass contract and
  known-answer tests; interchangeability is not assumed from API compatibility.
- If all eligible providers are unavailable, persist work and explain the actual
  reason: quota exhaustion, temporary outage, invalid configuration, or unsupported
  output. Communicate whether retry/resume is possible without exposing secrets,
  promising an unknown reset time, or replacing computed evidence with guesses.
- Jev integration, keys, review calls, and provider dependencies are excluded.
- Preserve old chats as readable legacy conversations. New investigations have
  explicit analytical state; avoid rewriting historical message metadata.

### Analysis and agent boundaries

- Profile a dataset deterministically at ingestion. A compact profile supplies
  the planner with types, candidate metrics/dates, quality signals, and bounded
  category examples; exclude unnecessary raw rows from model requests.
- The planner returns a schema-validated plan or a targeted clarification.
  It selects allowlisted operations, not arbitrary code or SQL expressions.
- Initial tools cover filters, aggregate/group-by, sorting/top values, period
  comparisons, contributions, and chart-series construction. Add correlation
  only with explicit sample and interpretation constraints.
- Tool validation enforces ownership, allowed columns/operators, supported
  types, step count, cardinality, output limits, and computational budgets.
  Initial proposal: at most four analysis steps per normal investigation.
- The graph controls execution and one bounded plan-repair attempt. Profiling,
  validation, chart creation, and orchestration are deterministic nodes; they
  are not presented as additional LLM agents.
- The analyst receives computed evidence and limitations, not unrestricted
  dataset access. It returns findings linked to evidence identifiers. Render
  authoritative numeric facts from stored evidence; validate identifiers and
  claim scope. Structural checks do not guarantee semantic truth.
- Failed computation produces a clear failure, never fabricated conclusions.
  If interpretation fails, preserve the successful evidence with an
  “interpretation unavailable” state and optional later retry within budget.
- Deprecate unrestricted generated Python on the new investigation path.
  A killable subprocess can improve timeout control in legacy execution but is
  not a security sandbox. Public arbitrary Python remains outside the new scope.
- Period calculations define date parsing/timezone, inclusive boundaries,
  incomplete-period behavior, missing-data policy, denominator handling, and
  category contributions before computing. Ask when assumptions change meaning.

### Contracts and persistence

- Introduce versioned contracts for dataset profile, analysis plan, evidence,
  findings, chart specifications, investigations, and run events. Validate them
  server-side and keep frontend types aligned.
- Separate a saved investigation from each run. Record dataset version, plan,
  selected context, evidence, model/prompt/tool versions, and execution timings.
- Add private object storage for CSV bytes and larger artifacts. Use owner-scoped
  paths/policies; client input must not grant access to another user's object.
- Store records, quota state, leases, and graph checkpoints durably in Postgres.
  Validate compatibility with the selected Supabase connection/pool mode.
  Keep checkpoint state small; reference evidence/artifacts instead of embedding
  entire DataFrames. Prune obsolete checkpoints/artifacts with a retention policy.
- Add forward migrations for existing databases, including one-dataset-per-chat
  enforcement after auditing existing duplicates. Do not rely on re-running the
  initial create-if-absent migration to alter old tables.
- Enforce ownership on every dataset, investigation, run, resume, result, event,
  and download operation. Backend service-role access does not replace this.

### Free-tier execution and quota behavior

- Target two model requests on the normal path: plan plus interpretation.
  Allow one additional request for repair. Account for all network attempts,
  including failures, in a separate bounded attempt budget; SDK retry defaults
  must not create hidden attempts. Budget exhaustion pauses or fails visibly.
  Proposed starting cap: two HTTP attempts per inference stage, at most three
  stages including repair (six total attempts); measure and tune conservatively.
  Failover consumes this same cap rather than introducing a second retry loop.
- Concept explanations use one bounded call. Dataset profiling, tool execution,
  known template reruns, and validated cache reuse require no model inference.
- Persist project/model-level scheduling and estimated token reservations.
  Configure limits from the owner's active AI Studio quota; actual capacity is
  not inferred from a key or generic documentation. Multiple keys in a Google
  project share quota. Honor RetryInfo/Retry-After and bounded jitter/backoff.
- Track daily limits against Pacific quota reset, including daylight-saving
  changes; display waits in the user's local timezone. A timer is not proof of
  recovered capacity. Usage outside this app can consume the same quota.
- Persist a run before executing it. Use idempotent step commits, lease ownership,
  and expiry so retries/restarts do not create duplicate results. If a provider
  response was lost before persistence, record the uncertainty; exactly-once
  external inference cannot be promised without provider support.
- Return a run identifier quickly. The browser polls actual persisted status
  while active. Add streaming only after durable event behavior works; reconnect
  can fall back to polling.
- Render executes work only while awake. Recover stale leases on startup and
  resume eligible work on an active request. Persist long quota waits rather
  than keeping a worker asleep. No guaranteed unattended execution on the free
  sleeping backend and no keep-alive workaround.
- Begin with one computational worker and conservative upload/row/output limits,
  then measure peak memory and latency before raising them. The existing 50 MB
  ceiling is not inherited as a guaranteed safe analysis limit.
- Cache computations using owner scope, immutable dataset version, normalized
  plan and tool version. Cache interpretation separately using evidence,
  contextual state, prompt/model versions. Do not cache transient failures as
  successful answers. Avoid obligatory vector indexing for this use case.

### User experience

- Left navigation: datasets and investigations. Main canvas: question, plan,
  findings, tables/charts. Collapsible evidence panel: filters, assumptions,
  coverage, and execution details. On small screens use sequential views.
- Use a restrained data-focused visual system, readable tables, deliberate
  typography, accessible focus states, and limited motion. Design the actual
  empty/error/wait states before polishing the marketing page.
- Show dataset quality immediately. Provide descriptive investigation starters.
  Complex/ambiguous plans can be edited or confirmed; avoid approval friction
  for unambiguous low-cost operations.
- Replace model confidence badges with concrete quality/coverage indicators.
  Keep measured observations and explanatory hypotheses distinguishable.
- Use bounded JSON chart specifications rendered interactively in the frontend.
  Preserve both tabular and graphical evidence. Legacy PNG charts remain readable.
- Saved runs show true states: queued, planning, needs clarification, computing,
  interpreting, waiting for quota, interrupted, complete, failed, cancelled.
  These are observable product states, not simulated stage timers.
- Subscribe to session refresh and handle expired authentication explicitly.
  Show load failures and retry actions rather than silently empty workspaces.

### Delivery sequence and acceptance gates

| Phase | Deliverable | Gate before progressing |
| --- | --- | --- |
| 0: baseline | Offline behavioral harness; routing, contextual cache, chart/evaluator, logging/health/session regressions; dependency audit | Known failures reproduced then resolved; types/syntax/build pass |
| 1: dataset/evidence | Private durable CSVs, profiles, deterministic tools, versioned evidence and chart contracts | Known-answer tool results match fixtures; ownership and restart file recovery pass |
| 2: vertical slice | One saved period-change investigation with Gemini planner/analyst and explicit graph | Mocked success/clarification/failure paths work; normal path uses two calls and repair stays within cap |
| 3: resilience | Persistent checkpoints, leases, quota scheduler, resume and real status events | 429, lost response, restart, duplicate submit, and exhausted-budget tests retain completed evidence |
| 4: workspace UI | Dataset browser, plan/results canvas, evidence panel, accessible states and downloads | Confirmed browser journeys pass at desktop/narrow widths; evidence navigation and session refresh work |
| 5: evaluation/release | Known-answer investigation benchmark, resource measurements, configuration/runbook and staged deployment | Measured quality/latency/memory fit actual quota; explicit deploy request before production mutation |

Each phase should remain independently reviewable. Use additive schema/API changes
and a feature switch for the new path; test legacy read behavior throughout.

## Testing Decisions

- Owner-confirmed primary seam: authenticated public upload/investigation API,
  with deterministic fake Gemini, database/storage boundaries, and controlled
  clock/queue inputs. Assert observable results, authorization, lifecycle state,
  call counts, and preserved evidence rather than internal graph node names.
- Existing chat/upload API and orchestrator boundary provide prior architectural
  seams. There is no checked-in behavioral test suite to copy; establish one.
- Add real local database integration coverage for migrations, RLS, uniqueness,
  atomic leases, and checkpoint transactions; mocks cannot prove those properties.
- Use small fixtures with independently calculated expected values: dates across
  month boundaries, incomplete periods, zero prior-period totals, missing values,
  duplicate rows, absent segments, high cardinality, and empty results.
- Verify concept/off-topic routing, changed-context follow-ups, cache invalidation,
  plot-plus-table evidence, unsupported plans, and failure without invented output.
- Simulate 429/daily exhaustion, transient error, permanent error, expired token,
  restart during a step, and duplicate submission. Confirm no repeated completed
  steps, retry caps, lease recovery, and clear partial-output behavior.
- Owner-confirmed browser journeys: sign-in/session refresh; upload/profile;
  clarification; successful investigation; evidence inspection; quota wait/resume;
  legacy conversation reading. Mock inference to avoid quota spending.
- Include keyboard/focus, narrow-screen layout, long table labels, and real
  loading/empty/error states in UI checks.
- Live Gemini smoke validation is a separate small-budget integration step. It
  checks model access and structured outputs, not general analytical quality.
- Benchmark on fixed known-answer questions. Report calculation correctness,
  unsupported-claim rate, clarification behavior, calls/tokens, queue wait versus
  inference latency, peak memory, and recovery outcomes. Report measurements,
  not invented targets presented as achieved scores.

### Demo readiness

- Exercise real deployed sign-in, upload/profile, clarification, investigation,
  evidence inspection, download, and saved-run reload before calling the release
  demo-ready. Verify both providers with a small live request budget.
- Test Gemini-to-Groq and Groq-to-Gemini failover, and all-provider exhaustion.
  Users see a clear reason and retain completed evidence in each failure case.
- Verify cold-start messaging, browser reconnection/session refresh, persisted
  dataset recovery, and restart during a run. A sleeping free backend cannot
  promise continuous execution, so interrupted work must resume visibly.
- Keep a tested sample dataset and saved completed investigation available for
  demos. Label saved results as saved, not fresh inference. Reserve/check actual
  quota before a live demo; cached evidence is useful but not a live-provider test.

### GitHub delivery

- Only `Shreyas-Sarkar` may perform GitHub writes for this repository. Verify
  authenticated login and remote immediately before pushing or other mutations.
- Deliver gradual coherent commits: repo setup, baseline fixes, storage/tools,
  provider integration, orchestration/recovery, and UI can be separate increments.
- Push each increment only after its relevant checks pass. Failed or unverified
  work stays local. Report which evidence is offline versus live; commits do not
  imply production deployment is verified or authorized.

## Out of Scope

- Jev; paid model fallback; paid hosting; hosted LangGraph platform; autonomous
  debate/reviewer swarms; unbounded tool loops or inference retries.
- Public execution of arbitrary Python/SQL; a regex or subprocess advertised
  as a secure sandbox.
- Database connectors, multi-dataset joins, forecasting, causal inference,
  enterprise collaboration, recurring unattended investigations, PDF reports.
- Guaranteed background completion while Render sleeps; quota circumvention.
- Implementing or deploying the redesign during repository setup/planning.

## Further Notes

Gemini key presence and GitHub access were confirmed; no live inference was
performed in this setup. Exact active model quota, private storage allowance,
Supabase connection constraints, and Render memory must be checked during the
corresponding phase. Existing files may already be missing after past restarts;
storage migration cannot reconstruct them and must offer re-upload.

References checked on 2026-10-02:

- [Gemini model](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)
- [Gemini quotas](https://ai.google.dev/gemini-api/docs/rate-limits): per-project
  limits and Pacific daily reset; account-specific values are authoritative.
- [Structured outputs](https://ai.google.dev/gemini-api/docs/structured-output):
  schema support still requires application-level semantic validation.
- [Render free services](https://render.com/docs/free): sleep and temporary files.
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence):
  durable checkpointer selection and bounded checkpoint retention.
