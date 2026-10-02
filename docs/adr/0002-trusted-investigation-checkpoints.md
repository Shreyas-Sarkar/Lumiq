# Trusted investigations with application checkpoints

Date: 2026-10-02
Status: Accepted

## Context

The existing generated-Python pipeline exposes arbitrary execution inside the API
process. Render free disk and memory are temporary; free providers frequently
reject requests. We need bounded analytical operations, retained evidence, and
recoverable steps without a continuously running worker.

## Decision

Use a self-hosted Python LangGraph with planner, trusted tools, and analyst nodes.
Only planner/analyst call models. Gemini native and Groq share schema validation
and the same attempt budget. New investigations never execute generated code.
Legacy messages remain readable; legacy execution is disabled by default.

Use application checkpoints in `investigation_run.state` through transactional
PostgREST RPCs, rather than LangGraph's built-in PostgresSaver. Reconstruct the
small graph from saved plan/evidence at recovery. Keep dataframes outside the
state. A saved investigation ID groups immutable completed runs and revisions.

This avoids adding a direct database password/session-pool connection and allows
the existing backend service-role client to persist state. SQL serializes run and
compute leases, and fences checkpoint writes by lease token/expiry. The browser
polls stored state; startup recovers up to three unfinished runs, and active reads
resume eligible unfinished work. Quota waits require explicit resume.

## Consequences

This is durable stage recovery, not native LangGraph checkpoint replay/time travel.
There is no guarantee of exactly-once external inference after a lost response.
An attempt is recorded before dispatch; a lost response remains unconfirmed and
consumes the budget. A stale worker cannot overwrite a newer lease holder.

Events are bounded to the most recent thirty; provider attempts to six. Evidence
reuse requires matching owner, dataset version, validated plan, and tool version.
Saved findings are not reused across changed questions/context. There is no
background-completion guarantee while Render sleeps. Validation checks shape,
references, and numerical claims; it cannot prove every qualitative observation.
