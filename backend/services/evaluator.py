from models.schemas import ExecutionResult, EvaluationResult


class Evaluator:
    def __init__(self, llm_client, max_retry_count: int = 2):
        self.llm_client = llm_client
        self.max_retry_count = max_retry_count

    async def evaluate(
        self, query: str, code: str, result: ExecutionResult
    ) -> EvaluationResult:
        if not result.success:
            return EvaluationResult(
                is_valid=False,
                is_relevant=False,
                confidence=0.0,
                reason=result.error or "Execution failed",
                should_retry=True,
                suggested_fix=result.error,
            )

        if result.output is None:
            return EvaluationResult(
                is_valid=False,
                is_relevant=False,
                confidence=0.0,
                reason="No result was produced",
                should_retry=True,
                suggested_fix="Ensure the variable 'result' is assigned a value",
            )

        import pandas as pd
        if isinstance(result.output, pd.DataFrame) and len(result.output) == 0:
            return EvaluationResult(
                is_valid=True,
                is_relevant=True,
                confidence=0.7,
                reason="Query returned empty DataFrame",
                should_retry=False,
            )

        try:
            llm_check = await self._check_relevance_with_llm(
                query, result.serialized_output[:300]
            )
        except Exception:
            llm_check = True

        return EvaluationResult(
            is_valid=True,
            is_relevant=llm_check,
            confidence=0.9 if llm_check else 0.4,
            reason="Result validated" if llm_check else "Result may not answer the question",
            should_retry=not llm_check,
        )

    async def _check_relevance_with_llm(
        self, query: str, result_preview: str
    ) -> bool:
        try:
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Does this result: {result_preview}\n"
                        f"answer this question: {query}\n"
                        "Reply with only: YES or NO"
                    ),
                }
            ]
            response = await self.llm_client.complete(
                messages, temperature=0.0, max_tokens=10
            )
            return "yes" in response.lower()
        except Exception:
            return True

    def should_retry(
        self, eval_result: EvaluationResult, attempt: int
    ) -> bool:
        return eval_result.should_retry and attempt < self.max_retry_count

    def generate_retry_prompt_suffix(self, code: str, error: str) -> str:
        return (
            f"\n\nThe previous code had this error: {error}\n"
            f"Previous code:\n```python\n{code}\n```\n"
            "Fix the error and write corrected code."
        )
