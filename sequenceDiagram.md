# Lumiq — Sequence Diagrams

## 1. Dataset Upload Lifecycle

```mermaid
sequenceDiagram
    actor User
    participant Frontend
    participant UploadRouter as POST /upload (FastAPI)
    participant DatasetManager
    participant RAGService
    participant Supabase
    participant ChromaDB
    participant FileSystem

    User->>Frontend: Selects CSV file in UploadModal
    Frontend->>Frontend: Validates file (type=CSV, size < 50MB)
    Frontend->>UploadRouter: POST /upload (multipart form: file, chat_id, Authorization: Bearer token)

    UploadRouter->>UploadRouter: Verify JWT token → extract user_id
    UploadRouter->>UploadRouter: Check chat belongs to user_id (Supabase lookup)

    UploadRouter->>FileSystem: Save file to /data/uploads/{user_id}/{dataset_id}.csv
    UploadRouter->>DatasetManager: load_csv(file_path)
    DatasetManager-->>UploadRouter: df: DataFrame

    UploadRouter->>DatasetManager: extract_schema(df)
    DatasetManager-->>UploadRouter: schema_info: dict

    UploadRouter->>DatasetManager: extract_sample_rows(df, n=5)
    DatasetManager-->>UploadRouter: sample_rows: list[dict]

    UploadRouter->>DatasetManager: extract_summary_stats(df)
    DatasetManager-->>UploadRouter: summary_stats: dict

    UploadRouter->>Supabase: INSERT into dataset table (all metadata)
    Supabase-->>UploadRouter: dataset_id: str

    UploadRouter->>Supabase: UPDATE chat SET dataset_id = dataset_id

    UploadRouter->>RAGService: get_or_create_collection(collection_id=dataset_id)
    RAGService->>ChromaDB: Create named collection
    ChromaDB-->>RAGService: collection

    UploadRouter->>RAGService: index_dataset(collection_id, schema, samples, stats)
    RAGService->>RAGService: build_documents() → 3 documents [schema_doc, sample_doc, stats_doc]
    RAGService->>ChromaDB: collection.add(documents=docs, embeddings=embed(docs), ids=[...])
    ChromaDB-->>RAGService: OK

    UploadRouter-->>Frontend: 200 OK {dataset_id, filename, row_count, column_count, schema_preview}
    Frontend-->>User: Shows dataset preview card in chat
```

---

## 2. Query Handling Lifecycle (Core Pipeline)

```mermaid
sequenceDiagram
    actor User
    participant Frontend
    participant ChatRouter as POST /chat (FastAPI)
    participant Orchestrator as ChatOrchestrator
    participant ChatManager
    participant DatasetManager
    participant RAGService
    participant CodeGenerator
    participant ExecutionEngine
    participant Evaluator
    participant ExplanationService
    participant LogService
    participant Supabase
    participant ChromaDB
    participant LLMClient
    participant LLM_API as Groq / Gemini API

    User->>Frontend: Types query, presses Send
    Frontend->>Frontend: Sets UI state → "thinking" (shows phase 1 "Understanding dataset…")
    Frontend->>ChatRouter: POST /chat {chat_id, query, Authorization: Bearer token}

    ChatRouter->>ChatRouter: Verify JWT → extract user_id
    ChatRouter->>Orchestrator: handle_query(chat_id, user_id, query)

    Note over Orchestrator: Phase 1 — Load context

    Orchestrator->>ChatManager: add_message(chat_id, role="user", content=query)
    ChatManager->>Supabase: INSERT message (role=user)
    Supabase-->>ChatManager: message_id

    Orchestrator->>DatasetManager: get_dataset_by_chat_id(chat_id)
    DatasetManager->>Supabase: SELECT dataset WHERE chat_id = chat_id
    Supabase-->>DatasetManager: Dataset (metadata, storage_path, chroma_collection_id)
    DatasetManager-->>Orchestrator: dataset

    Orchestrator->>DatasetManager: load_dataframe_from_storage(dataset.storage_path)
    DatasetManager-->>Orchestrator: df: DataFrame

    Orchestrator->>ChatManager: get_recent_context(chat_id, n=10)
    ChatManager->>Supabase: SELECT last 10 messages ordered by sequence_number DESC
    Supabase-->>ChatManager: list[Message]
    ChatManager-->>Orchestrator: chat_history: list[dict]

    Note over Orchestrator: Phase 2 — RAG Retrieval

    Orchestrator->>RAGService: retrieve_context(collection_id=dataset.chroma_collection_id, query=query, n_results=3)
    RAGService->>ChromaDB: collection.query(query_texts=[query], n_results=3)
    ChromaDB-->>RAGService: relevant_docs: list[str]
    RAGService-->>Orchestrator: schema_context: list[str]

    Note over Orchestrator: Phase 3 — Code Generation

    Orchestrator->>CodeGenerator: generate_code(query, schema_context, chat_history, dataset.schema_info)
    CodeGenerator->>CodeGenerator: build_prompt(query, schema_context, history, schema)
    CodeGenerator->>LLMClient: complete(messages=[system_prompt, user_prompt])
    LLMClient->>LLM_API: POST /chat/completions {model, messages, temperature=0.1}
    LLM_API-->>LLMClient: LLM response text
    LLMClient-->>CodeGenerator: raw_response: str
    CodeGenerator->>CodeGenerator: extract_code_from_response(raw_response) — strip markdown fences
    CodeGenerator->>CodeGenerator: validate_code_safety(code) — check forbidden patterns
    CodeGenerator-->>Orchestrator: CodeGenerationResult {code, raw_response}

    Note over Orchestrator: Phase 4 — Execution

    Orchestrator->>ExecutionEngine: execute(code=code, df=df)
    ExecutionEngine->>ExecutionEngine: build_safe_globals() — restricted __builtins__, inject df + allowed libs
    ExecutionEngine->>ExecutionEngine: _check_forbidden_patterns(code) — final safety gate
    ExecutionEngine->>ExecutionEngine: _run_with_timeout(code, scope, timeout=10s)
    ExecutionEngine-->>Orchestrator: ExecutionResult {success, output, output_type, error, exec_time_ms}

    Note over Orchestrator: Phase 5 — Evaluation

    Orchestrator->>Evaluator: evaluate(query, code, execution_result)
    Evaluator->>Evaluator: _check_result_non_null(result)
    Evaluator->>Evaluator: _check_result_type_valid(result)
    Evaluator->>LLMClient: complete([relevance_check_prompt])
    LLMClient->>LLM_API: POST /chat/completions
    LLM_API-->>LLMClient: relevance verdict
    LLMClient-->>Evaluator: verdict: str
    Evaluator-->>Orchestrator: EvaluationResult {is_valid, is_relevant, should_retry, reason}

    alt should_retry == true AND retry_count < 2
        Note over Orchestrator: RETRY LOOP (max 2 times)
        Orchestrator->>CodeGenerator: generate_code(retry_prompt_with_error_context)
        CodeGenerator-->>Orchestrator: CodeGenerationResult (new code)
        Orchestrator->>ExecutionEngine: execute(new_code, df)
        ExecutionEngine-->>Orchestrator: new ExecutionResult
        Orchestrator->>Evaluator: evaluate again
        Evaluator-->>Orchestrator: new EvaluationResult
    end

    alt persistent failure after 2 retries
        Orchestrator->>ExplanationService: format_error_response(query, last_error, retry_count)
        ExplanationService-->>Orchestrator: rejection AssistantResponse
    end

    Note over Orchestrator: Phase 6 — Explanation

    Orchestrator->>ExplanationService: format_response(query, code, exec_result, eval_result, schema_context)
    ExplanationService->>ExplanationService: format_dataframe_result(df_output) if result is DataFrame
    ExplanationService->>ExplanationService: generate_reasoning(query, code, result, context)
    ExplanationService->>ExplanationService: extract_assumptions(code, schema)
    ExplanationService-->>Orchestrator: AssistantResponse {answer, code, result, reasoning, assumptions}

    Note over Orchestrator: Persist + Log

    Orchestrator->>ChatManager: add_message(chat_id, role="assistant", content=answer, metadata=full_response)
    ChatManager->>Supabase: INSERT message (role=assistant, metadata=jsonb)
    Supabase-->>ChatManager: assistant_message_id

    Orchestrator->>LogService: log_execution(chat_id, assistant_message_id, user_id, query, code, output, status, error, retry_count, exec_time)
    LogService->>Supabase: INSERT execution_log
    Supabase-->>LogService: OK

    Orchestrator-->>ChatRouter: ChatResponse {message_id, response: AssistantResponse, chat_id, created_at}
    ChatRouter-->>Frontend: 200 OK JSON

    Frontend->>Frontend: Renders AssistantResponse:
    Frontend->>Frontend:   → Answer text bubble
    Frontend->>Frontend:   → CodeBlock with syntax highlighting
    Frontend->>Frontend:   → DataTable or scalar result
    Frontend->>Frontend:   → Reasoning accordion
    Frontend->>Frontend:   → Assumptions list
    Frontend-->>User: Full response visible, UI state → "idle"
```

---

## 3. Authentication Flow

```mermaid
sequenceDiagram
    actor User
    participant Frontend
    participant Supabase_Auth as Supabase Auth

    User->>Frontend: Opens /login, enters email + password
    Frontend->>Supabase_Auth: supabase.auth.signInWithPassword({email, password})
    Supabase_Auth-->>Frontend: {session: {access_token, refresh_token, user}}
    Frontend->>Frontend: Store session in localStorage + Supabase client
    Frontend->>Frontend: Redirect to /chat

    Note over Frontend: On every API call:
    Frontend->>Frontend: Get current session token: supabase.auth.getSession()
    Frontend->>ChatRouter: Request with header: Authorization: Bearer {access_token}
    ChatRouter->>ChatRouter: Verify JWT with Supabase JWT secret
    ChatRouter->>ChatRouter: Extract user_id from JWT claims
    ChatRouter-->>Frontend: Authorized response
```

---

## 4. Load Past Chat Flow

```mermaid
sequenceDiagram
    actor User
    participant Frontend
    participant ChatRouter as GET /chats + GET /chat/{id}
    participant ChatManager
    participant Supabase

    User->>Frontend: Opens /chat (sidebar loads)
    Frontend->>ChatRouter: GET /chats (Authorization: Bearer token)
    ChatRouter->>ChatRouter: Extract user_id from JWT
    ChatRouter->>ChatManager: get_chats_by_user(user_id)
    ChatManager->>Supabase: SELECT * FROM chat WHERE user_id = uid ORDER BY updated_at DESC
    Supabase-->>ChatManager: list[Chat]
    ChatManager-->>ChatRouter: list[Chat]
    ChatRouter-->>Frontend: 200 OK [{id, title, updated_at, status}]
    Frontend-->>User: Sidebar shows chat list

    User->>Frontend: Clicks on a past chat
    Frontend->>ChatRouter: GET /chat/{chat_id}
    ChatRouter->>ChatManager: get_chat_by_id(chat_id, user_id)
    ChatManager->>Supabase: SELECT chat + messages WHERE chat_id = id AND user_id = uid
    Supabase-->>ChatManager: Chat + list[Message]
    ChatManager-->>ChatRouter: {chat, messages, dataset_preview}
    ChatRouter-->>Frontend: 200 OK full chat data
    Frontend-->>User: Full conversation rendered, ready for new queries
```

---

## 5. Code Safety Validation Flow (ExecutionEngine detail)

```mermaid
sequenceDiagram
    participant CodeGenerator
    participant ExecutionEngine
    participant Orchestrator

    CodeGenerator->>ExecutionEngine: execute(code, df)

    ExecutionEngine->>ExecutionEngine: _check_forbidden_patterns(code)
    Note over ExecutionEngine: Blocks: import os, import sys,<br/>subprocess, open(, exec(, eval(,<br/>__import__, requests, socket,<br/>write, delete, drop

    alt Forbidden pattern found
        ExecutionEngine-->>Orchestrator: ExecutionResult(success=False, error="Unsafe code pattern: {pattern}")
    end

    ExecutionEngine->>ExecutionEngine: build_safe_globals()
    Note over ExecutionEngine: Allowed: pandas as pd,<br/>numpy as np, math, datetime,<br/>statistics, re, df (injected).<br/>__builtins__ = restricted dict<br/>(no __import__, no open, etc.)

    ExecutionEngine->>ExecutionEngine: _run_with_timeout(code, scope, timeout=10)
    Note over ExecutionEngine: Uses threading.Thread + join(timeout).<br/>If thread alive after 10s → killed

    alt Execution error
        ExecutionEngine-->>Orchestrator: ExecutionResult(success=False, error=traceback)
    end

    alt Execution timeout
        ExecutionEngine-->>Orchestrator: ExecutionResult(success=False, error="Execution timeout (10s)")
    end

    ExecutionEngine->>ExecutionEngine: serialize_result(output)
    Note over ExecutionEngine: DataFrame → {columns, rows, shape}<br/>Series → list<br/>scalar → value<br/>None → "No output returned"

    ExecutionEngine-->>Orchestrator: ExecutionResult(success=True, output=serialized, output_type=type_str)
```
