# Lumiq

A CSV-based AI analysis prototype. Upload a dataset, ask a question, and inspect
computed results, generated Python, charts, and model-written interpretations.

The pipeline is execution-first: exhausted computation failures return an error
instead of a model-generated replacement answer. Successful execution does not
guarantee the code answered the intended question or that its interpretation is
correct. The current runner is in-process, not an isolated execution sandbox.

## Current stack

- Frontend: Next.js, React, Tailwind CSS.
- Backend: FastAPI, Pandas, NumPy, Matplotlib/Seaborn.
- Accounts/records: Supabase Auth/Postgres.
- Dataset bytes: backend local filesystem.
- Model calls: configurable OpenAI-compatible chat-completions endpoint.
- Optional metadata retrieval: Qdrant and Cohere.

One CSV per chat. Default upload ceiling: 50 MB; processing capacity depends on
memory and concurrency. Progress labels are currently timer-driven, not live events.
Gemini inference and a multi-agent graph are planned, not implemented. Jev is
excluded from the planned architecture.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and server commands. Backend startup
requires Supabase configuration and `LLM_API_KEYS`; frontend needs its public
Supabase configuration and API URL. Start from the directory-specific environment
examples, preserving existing files. Qdrant/Cohere are optional.

After installing frontend dependencies, run from the root:

```sh
bash scripts/check.sh
```

This checks backend syntax and frontend types without provider calls. There is no
checked-in application test suite yet; live integrations need separate checks.

## AI-assisted development

- [AGENTS.md](AGENTS.md): concise project instructions.
- [CONTEXT.md](CONTEXT.md): current implementation map.
- [GLOSSARY.md](GLOSSARY.md): domain vocabulary.
- [Known limitations](docs/known-issues.md): observed gaps and validation limits.
- [Investigation workspace spec](.scratch/investigation-workspace/spec.md): phased redesign plan.
- [Local ticket workflow](docs/agents/issue-tracker.md).
- [Agent context decision](docs/adr/0001-project-local-agent-context.md).
- [Installed skills](.agents/skills/SOURCES.md): pinned Matt Pocock workflows with MIT attribution.

## Deployment constraints

The intended deployment is a Render free backend and Vercel frontend with
rate-limited model APIs. Current CSV bytes, cache, and request queue are temporary
process/disk state. They need durability work before reliable restart recovery.

Root `idea.md`, diagram files, and `vercel.json` are historical design/deployment
material. They retain obsolete ChromaDB and schema/mode references; use current
code and `CONTEXT.md` for implementation decisions. Checked-in deployment files
are not proof of the active production configuration.
