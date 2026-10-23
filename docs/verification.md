# Verification record

Backend/provider/SQL checks: 2026-10-02. Frontend and account checks: 2026-10-23.
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
  `oanvayxyzjpwqsndctgg` reports ACTIVE_HEALTHY. A read-only SQL query confirmed
  the new run table/private dataset bucket are still absent. The separate app
  connector targets another organization; use the authenticated personal CLI.
- Self-hosted Manrope font and license included; no browser request to a font CDN.
  Homepage sample uses local synthetic values and spends no provider quota.

## Local calculation benchmark

| Rows | Parsed data | Peak process RSS | Calculation time | Known answer |
| --- | --- | --- | --- | --- |
| 6 | <0.01 MB | 80.34 MB | 4.12 ms | Pass |
| 99,996 | 17.52 MB | 103.72 MB | 177.21 ms | Pass |

Measured on this development machine with `scripts/benchmark.py`. It measures
only the calculation process, not deployed end-to-end memory, latency, or model
quality. No model calls were made. No unsupported-claim score is claimed.

## Remaining live gates

Apply the reviewed additive migration, enable/configure the deployment, and run
`docs/runbook.md`'s live checklist. Deployed signup, private storage across restart,
worker recovery with real PostgREST, cold starts, and frontend/backend wiring
remain distinct from the successful offline and direct-provider checks.
