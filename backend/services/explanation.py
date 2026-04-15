import json
import pandas as pd
from models.schemas import ExecutionResult, EvaluationResult


class ExplanationService:
    """
    Formats the final response dict consumed by the orchestrator.

    Responsibilities:
    - generate_answer_summary:  concise factual answer from execution result
    - format_enriched_response: full response dict merging exec + cognitive output
    - format_error_response:    response dict for execution/LLM failures
    - format_rejection_response: response dict for dataset-not-found etc.
    """

    # ------------------------------------------------------------------
    # Primary formatter (success path)
    # ------------------------------------------------------------------

    def format_enriched_response(
        self,
        query: str,
        code: str,
        exec_result,
        cognitive_output: dict,
        rag_context: list[str],
        retry_count: int,
        mode: str,
    ) -> dict:
        """
        Build the full response payload combining a verified execution result
        with the cognitive engine's enrichment output.
        `exec_result` may be None in concept mode (no execution performed).
        """
        # For concept mode or any other no-execution path, generate the answer
        # directly from the cognitive insight rather than the execution result.
        if exec_result is not None:
            answer = self.generate_answer_summary(query, exec_result, mode)
        else:
            answer = cognitive_output.get("insight", "Concept explained.")

        # Deserialize the result for JSON transport
        serialized = exec_result.serialized_output if exec_result is not None else ""
        try:
            result_data = json.loads(serialized) if serialized else None
        except Exception:
            result_data = serialized or None

        # For plot results, include the base64 image in the result field
        if exec_result is not None and exec_result.output_type == "plot" and exec_result.plot_base64:
            result_data = {"plot_base64": exec_result.plot_base64}

        result_type = exec_result.output_type if exec_result is not None else "text"

        return {
            "answer": answer,
            "insight": cognitive_output.get("insight", ""),
            "anomalies": cognitive_output.get("anomalies", []),
            "follow_ups": cognitive_output.get("follow_ups", []),
            "confidence": cognitive_output.get("confidence", "medium"),
            "code": code,
            "result": result_data,
            "result_type": result_type,
            "mode": mode,
            "rag_context_used": rag_context,
            "retry_count": retry_count,
        }

    # ------------------------------------------------------------------
    # Answer summary generation
    # ------------------------------------------------------------------

    def generate_answer_summary(
        self, query: str, result: ExecutionResult, mode: str = "executor"
    ) -> str:
        """
        Produce a concise, factual 1–2 sentence answer from the execution result.
        For analyst mode this doubles as the primary answer (interpretation comes
        from the cognitive engine's insight field).
        """
        if not result or not result.success:
            return (
                f"I was unable to complete this analysis: "
                f"{result.error if result else 'unknown error'}"
            )

        if result.output_type == "plot":
            return "I've generated a visualization for your question."

        output = result.output

        if isinstance(output, pd.DataFrame):
            n = len(output)
            if n == 0:
                return "The query returned no matching results."
            if n == 1:
                # Single row – try to extract a scalar-like answer
                row = output.iloc[0].to_dict()
                pairs = ", ".join(
                    f"{k}: {v}" for k, v in list(row.items())[:4]
                )
                return f"Result: {pairs}."
            return f"Found {n:,} row{'s' if n != 1 else ''} matching your query."

        if isinstance(output, list):
            n = len(output)
            if n == 0:
                return "No items matched your query."
            return f"The result contains {n:,} item{'s' if n != 1 else ''}."

        if output is not None:
            return f"The result is: {str(output)}"

        return "Analysis complete."

    # ------------------------------------------------------------------
    # Error / rejection formatters
    # ------------------------------------------------------------------

    def format_error_response(
        self, query: str, error: str, retry_count: int
    ) -> dict:
        return {
            "answer": "I was unable to complete this analysis after multiple attempts.",
            "insight": f"The analysis encountered an error: {error}",
            "anomalies": [],
            "follow_ups": [],
            "confidence": "low",
            "code": "",
            "result": None,
            "result_type": "error",
            "mode": "executor",
            "rag_context_used": [],
            "retry_count": retry_count,
        }

    def format_rejection_response(self, query: str, reason: str) -> dict:
        return {
            "answer": "I'm unable to answer this question using the uploaded dataset.",
            "insight": reason,
            "anomalies": [],
            "follow_ups": [],
            "confidence": "low",
            "code": "",
            "result": None,
            "result_type": "rejection",
            "mode": "executor",
            "rag_context_used": [],
            "retry_count": 0,
        }

