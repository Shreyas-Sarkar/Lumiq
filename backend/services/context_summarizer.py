import logging

logger = logging.getLogger(__name__)


class ContextSummarizer:
    """
    Produces a short, one-sentence summary of what the user has analyzed
    so far in the current chat session.

    Uses a lightweight LLM call (max_tokens=60) when there are multiple
    prior queries. Falls back to deterministic string construction for
    zero or one prior query to avoid unnecessary LLM cost.

    Output example:
      "So far you've analyzed GPA trends, scholarship distribution, and
       attendance patterns."
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def summarize(self, messages: list[dict]) -> str:
        """
        Args:
            messages: Recent chat messages (role, content dicts).
        Returns:
            A short summary string for use in the cognitive engine prompt.
        """
        if not messages:
            return "No prior analysis in this session."

        # Extract user queries only (skip assistant turns)
        user_queries = [
            m["content"][:120]
            for m in messages
            if m.get("role") == "user"
        ]

        if not user_queries:
            return "No prior analysis in this session."

        if len(user_queries) == 1:
            return f"Previously asked: \"{user_queries[0]}\""

        # For 2+ queries, use LLM for a coherent sentence
        try:
            joined = "\n".join(f"- {q}" for q in user_queries[-6:])
            payload = [
                {
                    "role": "user",
                    "content": (
                        "In one short sentence starting with "
                        "'So far you\\'ve analyzed', summarize "
                        "what the user explored:\n"
                        f"{joined}"
                    ),
                }
            ]
            summary = await self.llm_client.complete(
                payload, temperature=0.1, max_tokens=60
            )
            return summary.strip().rstrip(".")  # clean trailing period
        except Exception as e:
            logger.warning("ContextSummarizer LLM call failed: %s", e)
            # Deterministic fallback
            topics = "; ".join(user_queries[-3:])
            return f"Previously analyzed: {topics}"
