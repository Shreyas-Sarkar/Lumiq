# Verification record

Offline backend/SQL checks: 2026-10-02. Live local-app verification: 2026-10-23.
Local results, not a production-readiness certification.

## Verified

- Clean backend dependency installation and `pip check` on Python 3.11 and 3.12.
- Twenty-one Python tests: known-answer period changes/contributions, zero baseline,
  invalid columns/windows, routing/cache/chart regressions, public investigation
  and upload APIs, ownership, idempotency, clarification, quota/failover, invalid
  model output, bounded repair, evidence reuse, recovery, CSV export, durable
  storage through a fake boundary, and foreign storage-path rejection.
- Public API tests use fake authenticated users and mocked provider/storage
  boundaries; no inference quota is spent by those tests.
- Fresh local PostgreSQL migration application and SQL checks for ownership RLS,
  lease exclusivity, denied foreign claims, checkpoint writes, quota reservations,
  and service-only RPC privileges. Local auth/storage schemas are test shims;
  actual Supabase storage policies still need the live checklist.
- Frontend typecheck and Next.js 16 production build. Production npm audit:
  zero reported vulnerabilities at verification time.
- Seven browser journeys passed on 2026-10-23: sign-in/create/upload/profile/results;
  evidence/download/mobile; clarification; quota resume; session refresh and legacy
  reads; editing period windows and filters; homepage sample exploration, source
  CSV download, keyboard entry, reduced motion, and workspace entry.
- Homepage checked at 320, 390, 768, 1024, and 1440 px without horizontal overflow.
  Mobile Lighthouse category scores: accessibility 100, best practices 100, SEO 100.
  Those scores exclude performance and do not certify deployed behavior.
- A separate local production-build trace at 390 px reported LCP 85 ms and
  CLS 0.00. This was an unthrottled loopback load, not a deployed or slow-network
  measurement; no field INP or real-user performance claim is made.
- Desktop and 390 px mobile screenshots inspected for clipping, readable tables,
  usable controls, and evidence navigation.
- Direct provider check on 2026-10-02, Gemini `gemini-3.5-flash-lite`: valid planner/analyst structured outputs.
- Direct provider check on 2026-10-02, Groq `openai/gpt-oss-20b`: valid planner/analyst structured outputs.
  The previous Groq model was unavailable; a model-list check confirmed the
  replacement before testing. This is access/schema validation, not a broad
  model-quality benchmark.
- Personal Supabase CLI access confirmed on 2026-10-23: Lumiq project
  `oanvayxyzjpwqsndctgg` reports ACTIVE_HEALTHY. After explicit owner approval,
  the additive investigation migration was applied atomically; duplicate dataset
  preflight was clear. Browser health reports database/workspace ready. The separate
  app connector targets another organization; use the authenticated personal CLI.
- Self-hosted Inter includes its OFL license; no browser request to a font CDN.
  Homepage sample uses local synthetic values and spends no provider quota.

## Live browser verification — 2026-10-23

Functional checks used the actual browser, hosted personal Supabase, real Gemini
and Groq, and the owner-supplied EDA practice CSV. No mocked responses were used
for this pass. The CSV and credentials are not committed.

- Correct login succeeds; an incorrect password shows an explicit error. Sign-out
  and a fresh login restore both workspaces, the dataset, and six completed runs.
- Two uploads each report 110 rows, eight columns, and eight retained duplicates.
  Missing values and date coverage are visible in the quality panel.
- Purchase amount total is 215,000; city groups sum to that total. An edited mean
  returns 2,529.412 across 85 usable amounts; a Mumbai filter returns 56,000.
- Half-year comparison returns 104,000 then 82,000, a decrease of 22,000
  (21.154%). Matching row counts increase from 43 to 52. Comparison statements
  now derive their direction from these measurements, including historical runs.
- Ambiguous periods trigger clarification. A real Groq planner fallback succeeded
  after invalid Gemini output. Invalid analyst output preserves evidence and offers
  bounded manual resume; the comparison completed within its six-attempt limit.
- Correlation returns approximately -0.018 for Age/Salary (81 pairs). Editing the
  second metric to Purchase_Amount returns approximately 0.193 (79 pairs), with
  a truthful revised title and the original run retained.
- A conceptual Pearson explanation completes without new calculations. Evidence
  inspection, CSV browser download, saved-run navigation, and profile-derived
  starter questions work. Narrow viewport inspection found no page overflow.
- Stopping the API shows an explicit local-backend message while saved results
  remain readable. Restart/retry restores access; repeated restarts recover data
  from private storage and persisted run state.
- Workspace rename saves independently for both workspaces and survives reload.
  Whitespace is trimmed, blank names cannot be saved, and cancelling preserves
  the existing name. The CSV and saved investigations remain intact.
- Browser access to the unauthenticated chats endpoint is rejected. This pass did
  not create a second account to re-test foreign ownership; that boundary retains
  the earlier mocked API and SQL coverage.

The supplied CSV validates these paths, not every possible analytical question.
No intentional real quota exhaustion, new-account signup, deployed cold-start test,
or production hosting rollout was performed. Model interpretation remains fallible.
Supabase security advisors still report existing GraphQL schema visibility,
public trigger-function reachability, and disabled leaked-password protection;
this pass is not a clean security-audit claim.

## Homepage refinement — 2026-10-23

After the live functional pass and workspace renaming, the homepage was refined
from the earlier layout using Linear/Fanout as visual references. The owner then
selected a closer Linear direction: near-black/off-white, Inter, restrained borders.
Shared tokens and the local font also apply to workspace, login, signup, and dialogs.
An original aperture mark replaces the boxed initial in all navigation/auth locations
and the SVG favicon. Browser inspection confirms the generated icon metadata.
Native CSS and a small IntersectionObserver/WAAPI leaf provide short, finite motion;
reduced motion cancels reveals and removes CSS animation/transforms. No animation
library, font CDN, or inference request is required for homepage interactions.

Browser checks cover all three sample views, source CSV download, all three trace
views, keyboard activation, section anchors, and workspace entry. Viewport checks
at desktop, tablet, and narrow/mobile sizes found no horizontal page overflow.
Login/signup were visually checked at a narrow viewport without creating an account;
input focus remains visible. The themed workspace retains its saved results and
rename dialog. A fresh page load after compilation produced no browser errors. The current browser
control does not expose reduced-motion emulation; that fallback was source-reviewed,
not re-certified by a live media-preference switch in this pass.

A subsequently revoked Supabase session returned `session_not_found`. The explicit
sign-in recovery path restored both renamed workspaces and saved results. This is
not evidence that every possible refresh/network interruption is resolved.

## Merge-fix verification — 2026-10-24

- The existing 21-test baseline exposed a read-time regression: provider failure
  was given completed-looking comparison findings. The original regression now
  passes, retaining evidence with an empty interpretation.
- Thirty-one mocked backend/API/tool tests pass. New red-before-green API cases cover
  all-missing grouped sums versus true zero, absent categories versus missing
  period measurements (including missing category labels), worker completion
  between read and lease claim for both cancel and resume, changed clarification
  after partial evidence, recovery of obsolete tool-version evidence, and suppression
  of model-generated rankings of groups whose measurements are unknown.
- Exact interpretation reuse completes an explicit-plan repeat with zero model
  calls. Changed question/context/model configuration misses the cache; malformed
  cached evidence references are rejected and fresh interpretation is requested.
  Private cache payload/key fields are excluded from public responses.
- `bash scripts/check.sh` passes Python syntax and frontend types. The Next.js
  production build passes. These static checks are separate from runtime coverage.
- Both standards and spec re-reviews found no remaining concrete blocker in the
  fix diff. Reviewers inspected source/tests; they did not independently rerun tests.
- Live browser follow-up: a ten-row synthetic CSV with unknown values, real zeros,
  absent categories, and missing category labels uploads and profiles successfully.
  A completed group-by run displays `MissingBoth` as unknown and `Zero` as 0; the
  unknown chart bar has zero width. A model-generated ranking of unknown results
  was reproduced, then suppressed with a deterministic limitation statement and
  its own red/green API regression. The same saved result replays correctly after
  backend restart. This live run used two successful Gemini requests; replay made
  no additional inference call. Interpretation cache hit/miss and completion-race
  timing are verified through the mocked API tests, not claimed as live race tests.
- No database migration or deployment setting is changed by these fixes. The cache
  uses existing owner-scoped persisted run JSON. Render/Vercel production behavior
  remains a separate release gate.

## Local calculation benchmark

| Rows | Parsed data | Peak process RSS | Calculation time | Known answer |
| --- | --- | --- | --- | --- |
| 6 | <0.01 MB | 80.34 MB | 4.12 ms | Pass |
| 99,996 | 17.52 MB | 103.72 MB | 177.21 ms | Pass |

Measured on this development machine with `scripts/benchmark.py`. It measures
only the calculation process, not deployed end-to-end memory, latency, or model
quality. No model calls were made. No unsupported-claim score is claimed.

## Remaining live gates

The personal Supabase migration and local live flows are now verified. Configure
and deploy Render/Vercel, then run the runbook's checklist against those deployed
origins. New-account signup, cold starts, and deployed frontend/backend wiring
remain unverified. Earlier automated browser journeys used mocked auth/API
boundaries; their accessibility/performance results predate this live-fix pass.
