# Known limitations

Source review and local checks on 2026-10-02. These are observations, not proof
of a deployed incident or permission to change every subsystem.

| Area | Evidence / consequence | Relevant backend service unless qualified |
| --- | --- | --- |
| Execution | Regex-filtered in-process `exec()`. Timeout leaves the daemon thread running. | `execution_engine.py` |
| Routing | “What is correlation in general?” and “What is the weather today?” route to executor. Column matches can override hybrid intent. | `mode_classifier.py` |
| Validation | Final executor acceptance can ignore negative relevance evaluation; hybrid skips relevance evaluation. | `orchestrator.py`, `evaluator.py` |
| Charts | Plot-only success has `output=None` and triggers evaluator retries. Charts with explicit results are omitted by formatter. | `evaluator.py`, `explanation.py` |
| Interpretation | Prompts do not enforce semantic correctness. Confidence is model-generated; evidence text is truncated. | `cognitive_engine.py` |
| Cache | Keys omit conversation state; errors are cached. Invalidation searches for a dataset hash inside an already-hashed key. | `query_cache.py`, `orchestrator.py` |
| Durability | CSV bytes are local; database chats can outlive their files. | `dataset_manager.py`, `backend/routers/upload.py` |
| Concurrency | Read-max-then-insert message sequencing can collide. Parallel uploads can pass the dataset guard. | `chat_manager.py`, upload route, migration |
| Latency | Generation, relevance, interpretation, and later-turn summary add separate model calls. All share a serialized queue. | `llm_client.py`, `orchestrator.py` |
| Progress | Stage labels advance on a timer, not backend events. | `frontend/app/chat/[id]/page.tsx` |
| Session/error UX | Detail page captures a token at load without subscribing to refresh; some failures are swallowed. | Chat detail page |
| Index status | Failed indexing is tolerated but upload claims indexing succeeded. | Upload route, `rag_service.py` |
| Health/logs | Cache entry text participates in `all_ok`, causing degraded status. Logs omit errors and set elapsed time to zero. | `backend/main.py`, `orchestrator.py` |
| Dependencies | Next.js 14.2.3 predates security patches; unused auth helpers are deprecated. Verify official advisories before upgrading. | `frontend/package.json` |
| Contracts | Legacy `analyst`, `dataset_id`, and Chroma references remain in types/comments/docs. | Backend schemas, frontend types/API, diagrams |

Verified locally: TypeScript, Python syntax, scalar total, grouped table, chart
capture, invalid-column error, and regex-blocked import. Live signup, database
policies, provider inference, production hosting, and concurrency are separate checks.
