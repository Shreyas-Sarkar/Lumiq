# Remaining limitations

Updated 2026-10-24. Check `verification.md` for measured evidence.

- The personal Supabase migration and local live upload/recovery flows passed.
  Render/Vercel deployment, new-account signup, and cold starts remain release gates.
- Stage checkpoints retain plans/evidence across restarts but do not promise
  exactly-once inference when a provider response is lost before persistence.
- A request resumes work only while Render is awake. Long quota waits are saved
  and require explicit resume. A displayed retry time is not guaranteed capacity.
- Provider quota ceilings are conservative application limits. Account usage
  outside Lumiq is unknown. Groq organization limits are not multiplied by keys.
- Reference validation and numerical-claim rejection do not prove qualitative
  semantic correctness. Hypotheses remain explicitly unverified.
- Models receive a bounded profile and evidence. Very wide profiles or unusually
  verbose plans may exceed the prompt budget and fail with an explicit reason.
- CSV analysis is single-dataset and bounded; dates use ISO/UTC. Sparse observation
  dates do not establish complete period coverage. No forecasting or causal claims.
- Evidence displays at most fifty groups; truncation is disclosed. Group
  cardinality is capped at five hundred.
- Checkpoints retain thirty status events and six network attempts. Saved runs and
  private dataset objects remain until explicitly cleaned up; monitor free storage.
- Legacy local files may already be missing; durable storage cannot reconstruct
  them. Create a new workspace and re-upload. Historical conversations remain readable.
- Legacy generated-Python execution is disabled by default. Its in-process thread
  timeout/regex filtering is not a security sandbox. Old message sequencing may
  collide if that legacy execution path is explicitly re-enabled concurrently.
- Completed historical results remain immutable and may predate calculation fixes.
  Create a new revision to apply tool version 1.1; unfinished older evidence is
  recomputed during recovery. Planner calls remain uncached; interpretation reuse
  requires exact matching inputs and configured model versions.
- Root legacy diagrams and `idea.md` are historical design material.

Fixed in this branch: conceptual/comparison routing precedence; contextual cache
keys and invalidation; transient-error caching; plot-only acceptance and chart/table
transport; negative evaluator acceptance; fake progress; stale session tokens;
silent workspace errors; upload indexing claims; misleading health aggregation;
vulnerable frontend dependency baseline. New investigations replace unrestricted
execution and mandatory vector indexing with trusted analytical tools.

Merge review fixes on 2026-10-24: unknown grouped measurements no longer become
zero, resume/cancel retain freshly claimed checkpoints, changed clarifications
invalidate stale evidence, and validated interpretation reuse is owner scoped.
The earlier unconditional read-time comparison finding is limited to completed
or existing findings, so provider failure retains evidence without a synthesized
interpretation. See the regression coverage in `verification.md`.
