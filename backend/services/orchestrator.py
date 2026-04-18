import asyncio
import logging
from datetime import datetime, timezone
from models.schemas import QueryResponse, AssistantResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard-fail response builder — called on execution failure, NO LLM fallback
# ---------------------------------------------------------------------------

def _hard_fail_response(mode: str, last_code: str, last_error: str, retry_count: int, rag_context: list) -> dict:
    """
    Deterministic failure response. This is returned when execution fails
    after all retries. No LLM call is made. No reasoning is attempted.
    """
    return {
        "answer": "Execution failed. Cannot derive conclusions from data.",
        "insight": f"Code execution failed after {retry_count} retries: {last_error}",
        "anomalies": [],
        "follow_ups": [],
        "confidence": "low",
        "code": last_code,
        "result": None,
        "result_type": "error",
        "mode": mode,
        "rag_context_used": rag_context,
        "retry_count": retry_count,
    }


class ChatOrchestrator:
    """
    Executor-first 4-mode pipeline.

    Modes:
        executor   — direct computation; code generated and executed
        hybrid     — relationship/comparison; MUST compute all required metrics
        concept    — definitional query; explain + map to schema; NO execution
        irrelevant — off-topic; rejected immediately

    Pipeline (executor / hybrid):
        query
         → mode_classifier  (heuristic + schema-aware, no LLM)
         → cache check       (query + dataset_hash)
         → [RAG retrieval]   (always)
         → code → exec → eval loop (with retry)
         → cognitive_engine  (ONE LLM call: insight + follow-ups + confidence)
         → context_summarizer (batched with cognitive)
         → format response
         → cache store → log

    Architectural invariant:
        If execution fails → HARD FAIL. No LLM fallback. No guessing.
        concept mode      → NEVER calls code generator or execution engine.
    """

    def __init__(
        self,
        chat_manager,
        dataset_manager,
        rag_service,
        code_generator,
        execution_engine,
        evaluator,
        explanation_service,
        log_service,
        mode_classifier,
        cognitive_engine,
        context_summarizer,
        query_cache,
        max_retry_count: int = 2,
    ):
        self.chat_manager = chat_manager
        self.dataset_manager = dataset_manager
        self.rag_service = rag_service
        self.code_generator = code_generator
        self.execution_engine = execution_engine
        self.evaluator = evaluator
        self.explanation_service = explanation_service
        self.log_service = log_service
        self.mode_classifier = mode_classifier
        self.cognitive_engine = cognitive_engine
        self.context_summarizer = context_summarizer
        self.query_cache = query_cache
        self.max_retry_count = max_retry_count

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def handle_query(
        self, chat_id: str, user_id: str, query: str
    ) -> QueryResponse:
        # 1. Persist user message + maybe auto-title
        user_message = self.chat_manager.add_message(
            chat_id=chat_id, role="user", content=query
        )
        messages_so_far = self.chat_manager.get_messages_by_chat(chat_id)
        if len(messages_so_far) <= 2:
            title = self.chat_manager.auto_generate_title(query)
            self.chat_manager.update_title(chat_id, title)

        # 2. Load dataset metadata
        try:
            dataset = self.dataset_manager.get_dataset_by_chat_id(chat_id)
        except ValueError:
            response_data = self.explanation_service.format_rejection_response(
                query, "No dataset uploaded for this chat. Please upload a CSV first."
            )
            return self._store_and_return(chat_id, user_id, query, "", response_data, None)

        # 3. Load dataframe
        try:
            df = self.dataset_manager.load_dataframe_from_storage(
                dataset["storage_path"]
            )
        except FileNotFoundError:
            response_data = self.explanation_service.format_rejection_response(
                query, "Dataset file not found. Please create a new chat and re-upload."
            )
            return self._store_and_return(chat_id, user_id, query, "", response_data, None)

        schema = dataset.get("schema_info", {})
        summary_stats = dataset.get("summary_stats", {})
        storage_path = dataset.get("storage_path", "")

        # 4. Cache check
        dataset_hash = self.query_cache.compute_dataset_hash(storage_path)
        cached = self.query_cache.get(query, dataset_hash)
        if cached:
            logger.info("Cache hit for query: %s", query[:60])
            assistant_msg = self.chat_manager.add_message(
                chat_id=chat_id,
                role="assistant",
                content=cached["answer"],
                metadata=cached,
            )
            return self._build_response(chat_id, assistant_msg["id"], cached)

        # 5. Classify mode — schema-aware (df passed for column matching)
        mode = self.mode_classifier.classify(query, df=df)
        logger.info("Mode classified as: %s for query: %s", mode, query[:60])
        print("[PIPELINE] mode:", mode)

        # 6. Get recent context for cognitive engine (parallel with RAG)
        chat_history = self.chat_manager.get_recent_context(chat_id, n=10)
        collection_id = dataset.get("chroma_collection_id", dataset["id"])

        # 7. Run context summarization always; run RAG only for non-executor modes.
        # Vector context is optional and must never break query handling.
        summary_task = asyncio.create_task(
            self.context_summarizer.summarize(chat_history)
        )
        rag_context = []
        if mode != "executor":
            try:
                rag_context = await asyncio.to_thread(
                    self.rag_service.retrieve_context,
                    collection_id=collection_id,
                    query=query,
                    n_results=3,
                )
            except Exception as e:
                logger.warning("Optional RAG retrieval failed: %s", e)
                print("[ERROR]", e)
                rag_context = []
        else:
            print("[VECTOR] search skipped for executor")

        try:
            chat_summary = await summary_task
        except Exception as e:
            logger.warning("Context summarization failed: %s", e)
            print("[ERROR]", e)
            chat_summary = "No prior analysis in this session."

        # 8. Route to appropriate pipeline
        if mode == "executor":
            response_data = await self._run_executor_pipeline(
                query=query,
                df=df,
                schema=schema,
                summary_stats=summary_stats,
                rag_context=rag_context,
                chat_history=chat_history,
                chat_summary=chat_summary,
            )
        elif mode == "hybrid":
            response_data = await self._run_hybrid_pipeline(
                query=query,
                df=df,
                schema=schema,
                summary_stats=summary_stats,
                rag_context=rag_context,
                chat_history=chat_history,
                chat_summary=chat_summary,
            )
        elif mode == "concept":
            response_data = await self._run_concept_pipeline(
                query=query,
                df=df,
                schema=schema,
                summary_stats=summary_stats,
                rag_context=rag_context,
                chat_history=chat_history,
                chat_summary=chat_summary,
            )
        else:
            # mode == "irrelevant"
            response_data = self.explanation_service.format_rejection_response(
                query, "This question is outside the scope of the uploaded dataset."
            )

        # 9. Cache the result
        self.query_cache.set(query, dataset_hash, response_data)

        # 10. Persist + return
        return self._store_and_return(
            chat_id, user_id, query, dataset.get("storage_path", ""), response_data,
            None  # exec_result handled inside pipelines for logging
        )

    # ------------------------------------------------------------------
    # EXECUTOR PIPELINE
    # ------------------------------------------------------------------

    async def _run_executor_pipeline(
        self,
        query: str,
        df,
        schema: dict,
        summary_stats: dict,
        rag_context: list[str],
        chat_history: list[dict],
        chat_summary: str,
    ) -> dict:
        retry_suffix = ""
        last_code = ""
        last_error = ""
        exec_result = None
        retry_count = 0

        for attempt in range(self.max_retry_count + 1):
            # Code generation
            try:
                gen_result = await self.code_generator.generate_code(
                    query=query,
                    schema_context=rag_context,
                    chat_history=chat_history,
                    dataset_schema=schema,
                    retry_suffix=retry_suffix,
                    mode="executor",
                )
                code = gen_result.code
            except Exception as e:
                print("[ERROR]", e)
                return self.explanation_service.format_error_response(
                    query, str(e), retry_count
                )

            # Safety check
            is_safe, safety_reason = self.code_generator.validate_code_safety(code)
            if not is_safe:
                if attempt < self.max_retry_count:
                    retry_suffix = self.evaluator.generate_retry_prompt_suffix(
                        code, f"Code safety violation: {safety_reason}"
                    )
                    retry_count += 1
                    continue
                return self.explanation_service.format_rejection_response(
                    query, f"Generated code failed safety validation: {safety_reason}"
                )

            # Execution
            print("[EXECUTOR] running code")
            exec_result = self.execution_engine.execute(code, df)
            eval_result = await self.evaluator.evaluate(query, code, exec_result)

            last_code = code
            last_error = exec_result.error or ""

            if not self.evaluator.should_retry(eval_result, attempt):
                break

            retry_suffix = self.evaluator.generate_retry_prompt_suffix(
                code, exec_result.error or eval_result.reason
            )
            retry_count += 1

        # Success path: enrich with cognitive engine
        if exec_result is not None and exec_result.success:
            stats_summary = self.cognitive_engine.build_stats_summary(summary_stats)
            cognitive_output = await self.cognitive_engine.analyze(
                query=query,
                result_summary=exec_result.serialized_output,
                schema=schema,
                stats_summary=stats_summary,
                chat_summary=chat_summary,
                mode="executor",
            )
            return self.explanation_service.format_enriched_response(
                query=query,
                code=last_code,
                exec_result=exec_result,
                cognitive_output=cognitive_output,
                rag_context=rag_context,
                retry_count=retry_count,
                mode="executor",
            )

        # Hard fail — no LLM fallback, no reasoning without computation
        logger.error(
            "Executor pipeline HARD FAIL after %d retries. Error: %s",
            retry_count, last_error,
        )
        return _hard_fail_response("executor", last_code, last_error, retry_count, rag_context)

    # ------------------------------------------------------------------
    # HYBRID PIPELINE (relationship / comparison — MUST compute all metrics)
    # ------------------------------------------------------------------

    async def _run_hybrid_pipeline(
        self,
        query: str,
        df,
        schema: dict,
        summary_stats: dict,
        rag_context: list[str],
        chat_history: list[dict],
        chat_summary: str,
    ) -> dict:
        """
        Hybrid mode generates code to compute ALL required statistical evidence
        (correlations, groupby aggregations, distributions) then passes the
        verified result to the cognitive engine for interpretation.

        Hard-fail on execution failure — no LLM reasoning without data.
        """
        retry_suffix = ""
        last_code = ""
        last_error = ""
        exec_result = None
        retry_count = 0

        for attempt in range(self.max_retry_count + 1):
            try:
                gen_result = await self.code_generator.generate_code(
                    query=query,
                    schema_context=rag_context,
                    chat_history=chat_history,
                    dataset_schema=schema,
                    retry_suffix=retry_suffix,
                    mode="hybrid",
                )
                code = gen_result.code
            except Exception as e:
                print("[ERROR]", e)
                return self.explanation_service.format_error_response(
                    query, str(e), retry_count
                )

            is_safe, safety_reason = self.code_generator.validate_code_safety(code)
            if not is_safe:
                if attempt < self.max_retry_count:
                    retry_suffix = self.evaluator.generate_retry_prompt_suffix(
                        code, f"Code safety violation: {safety_reason}"
                    )
                    retry_count += 1
                    continue
                return self.explanation_service.format_rejection_response(
                    query, f"Generated code failed safety validation: {safety_reason}"
                )

            exec_result = self.execution_engine.execute(code, df)
            last_code = code
            last_error = exec_result.error or ""

            if exec_result.success:
                break

            retry_suffix = self.evaluator.generate_retry_prompt_suffix(
                code, exec_result.error or "execution failed"
            )
            retry_count += 1

        # Success path
        if exec_result is not None and exec_result.success:
            stats_summary = self.cognitive_engine.build_stats_summary(summary_stats)
            cognitive_output = await self.cognitive_engine.analyze(
                query=query,
                result_summary=exec_result.serialized_output,
                schema=schema,
                stats_summary=stats_summary,
                chat_summary=chat_summary,
                mode="hybrid",
            )
            return self.explanation_service.format_enriched_response(
                query=query,
                code=last_code,
                exec_result=exec_result,
                cognitive_output=cognitive_output,
                rag_context=rag_context,
                retry_count=retry_count,
                mode="hybrid",
            )

        # Hard fail — no LLM reasoning without computed evidence
        logger.error(
            "Hybrid pipeline HARD FAIL after %d retries. Error: %s",
            retry_count, last_error,
        )
        return _hard_fail_response("hybrid", last_code, last_error, retry_count, rag_context)

    # ------------------------------------------------------------------
    # CONCEPT PIPELINE (definitional — NO code generation, NO execution)
    # ------------------------------------------------------------------

    async def _run_concept_pipeline(
        self,
        query: str,
        df,
        schema: dict,
        summary_stats: dict,
        rag_context: list[str],
        chat_history: list[dict],
        chat_summary: str,
    ) -> dict:
        """
        Concept mode: explain the concept, map to actual schema columns, suggest
        a follow-up computation. MUST NOT call code_generator or execution_engine.
        """
        stats_summary = self.cognitive_engine.build_stats_summary(summary_stats)
        cognitive_output = await self.cognitive_engine.analyze(
            query=query,
            result_summary="[CONCEPT MODE — no execution performed]",
            schema=schema,
            stats_summary=stats_summary,
            chat_summary=chat_summary,
            mode="concept",
        )
        return self.explanation_service.format_enriched_response(
            query=query,
            code="",
            exec_result=None,
            cognitive_output=cognitive_output,
            rag_context=rag_context,
            retry_count=0,
            mode="concept",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _store_and_return(
        self,
        chat_id: str,
        user_id: str,
        query: str,
        storage_path: str,
        response_data: dict,
        exec_result,
    ) -> QueryResponse:
        assistant_msg = self.chat_manager.add_message(
            chat_id=chat_id,
            role="assistant",
            content=response_data["answer"],
            metadata=response_data,
        )

        if self.log_service:
            try:
                self.log_service.log_execution(
                    chat_id=chat_id,
                    message_id=assistant_msg["id"],
                    user_id=user_id,
                    query=query,
                    code=response_data.get("code", ""),
                    output=str(response_data.get("result", "")),
                    status=(
                        "success"
                        if response_data.get("result_type") not in ("error", "rejection")
                        else "error"
                    ),
                    error=None,
                    retry_count=response_data.get("retry_count", 0),
                    exec_time=0.0,
                )
            except Exception as e:
                logger.warning("Log service failed: %s", e)

        return self._build_response(chat_id, assistant_msg["id"], response_data)

    def _build_response(
        self, chat_id: str, message_id: str, response_data: dict
    ) -> QueryResponse:
        anomalies = response_data.get("anomalies", [])
        # Ensure anomalies are dicts (not Pydantic models) for serialization
        anomaly_objs = []
        for a in anomalies:
            if isinstance(a, dict):
                from models.schemas import AnomalyItem
                anomaly_objs.append(
                    AnomalyItem(
                        description=a.get("description", ""),
                        possible_explanation=a.get("possible_explanation", ""),
                    )
                )
            else:
                anomaly_objs.append(a)

        return QueryResponse(
            message_id=message_id,
            response=AssistantResponse(
                answer=response_data["answer"],
                insight=response_data.get("insight", ""),
                anomalies=anomaly_objs,
                follow_ups=response_data.get("follow_ups", []),
                confidence=response_data.get("confidence", "medium"),
                code=response_data.get("code", ""),
                result=response_data.get("result"),
                result_type=response_data.get("result_type", "error"),
                mode=response_data.get("mode", "executor"),
                rag_context_used=response_data.get("rag_context_used", []),
                retry_count=response_data.get("retry_count", 0),
            ),
            chat_id=chat_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
