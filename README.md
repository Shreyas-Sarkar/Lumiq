# Lumiq

Lumiq turns a CSV question into an inspectable calculation plan, computed evidence,
and a saved investigation. It uses two model specialists—planner and analyst—in a
self-hosted LangGraph, with deterministic Pandas tools between them.

- Profile missing values, duplicates, data types, and date coverage.
- Calculate aggregates, groups, period changes, contributions, and correlations.
- Inspect tables, charts, filters, assumptions, and evidence-linked observations.
- Clarify ambiguous questions, edit calculations, download CSV results, and revisit runs.
- Switch between Gemini and Groq within bounded attempts and persisted quota limits.
- Retain completed evidence when interpretation fails; resume from saved stages.
- Preserve historical chats as readable conversations. Generated Python execution
  is disabled by default on the new path.

## Stack and limits

Next.js 16 / React 19 frontend on Vercel; FastAPI / Python LangGraph backend on
Render; Supabase Auth, Postgres, and private Storage. No hosted agent subscription,
Jev dependency, or paid fallback. Maximum upload: 5 MB; analysis: 100,000 rows,
100 MB parsed data, four steps, and six provider attempts per run.

Free Render can sleep. Polling and startup recovery advance unfinished work while
the server is awake. Provider capacity cannot be guaranteed; failures show a reason
and preserve available evidence. Qualitative model interpretation still needs judgment.

## Run and verify

See [development instructions](CONTRIBUTING.md), [configuration and demo runbook](docs/runbook.md),
and [verification report](docs/verification.md). Fresh databases need the initial
schema and the additive investigation migration before enabling the new backend.

```sh
bash scripts/check.sh
bash scripts/test.sh
```

Browser journeys live in `frontend/tests/`; run `npm test` with the frontend running
on localhost:3000. Tests fake external providers and storage. [Sample CSVs](samples/README.md)
include independently calculated expected answers.

## Repository navigation

- [Agent instructions](AGENTS.md) and [current architecture](CONTEXT.md).
- [Vocabulary](GLOSSARY.md), [architecture decisions](docs/adr/), and [limitations](docs/known-issues.md).
- [Original implementation specification](.scratch/investigation-workspace/spec.md).
- [Skill provenance](.agents/skills/SOURCES.md).

Current implementation and local verification do not establish deployed demo
readiness. Follow the runbook's live release checks before demonstrating production.
