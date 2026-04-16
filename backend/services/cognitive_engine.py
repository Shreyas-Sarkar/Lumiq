import json
import re
import logging

logger = logging.getLogger(__name__)

COGNITIVE_SYSTEM_PROMPT = """You are a senior data analyst providing expert interpretation of data query results.

Your job: given a user's question AND the actual computed result, produce a structured JSON analysis.

STRICT RULES — FOLLOW EXACTLY:
1. NEVER hallucinate. Every claim you make must be traceable to the provided result or dataset context.
2. ONLY interpret computed results. Every statement must be grounded in actual values from the result.
3. If the result is insufficient to draw conclusions, state "Insufficient computed data" in the insight.
4. The insight must reference actual computed values/patterns — never write generic statements like "the data shows a trend".
5. Anomalies: only flag real, meaningful anomalies (0–2 max). Skip this entirely if nothing is notable.
6. Follow-ups: generate exactly 3 smart, actionable questions that extend the current analysis. Reference actual column names from the schema.
7. Do NOT repeat any question already seen in the conversation summary.
8. Confidence levels:
   - "high": result is complete, unambiguous, directly answers the query
   - "medium": result is partial or requires some interpretation
   - "low": result is empty, failed, or insufficient to draw conclusions
9. BANNED WORDS — you must NEVER use: "appears", "suggests", "likely", "might indicate",
   "seems", "could be", "possibly", "probably", "may indicate". Every claim must be backed
   by the provided computed result. Violations make the response unreliable.
10. In CONCEPT mode: explain the concept clearly, map it to actual column names from the schema
    where relevant, and suggest a concrete follow-up computation. Do NOT fabricate data values.

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown, no prose outside JSON:
{
  "insight": "1-2 sentences of dataset-specific interpretation referencing actual computed values",
  "anomalies": [
    {
      "description": "specific observation e.g. 'Column X has 34% null values'",
      "possible_explanation": "plausible reason grounded in the data"
    }
  ],
  "follow_ups": [
    "Question 1 referencing specific column names?",
    "Question 2 extending the analysis?",
    "Question 3 for deeper investigation?"
  ],
  "confidence": "high | medium | low"
}"""


# Sentinel prefix that marks a non-computed result — triggers hard guard
_NO_COMPUTE_SENTINEL = "[Grounding computation failed"


class CognitiveEngine:
    """
    Single LLM call that enriches a verified execution result with:
    - A dataset-specific, grounded insight
    - 0–2 real anomalies (only if meaningful)
    - Exactly 3 follow-up questions
    - Confidence level

    Called ONCE per query, AFTER the executor has produced a verified result.
    If no computed result is available the guard returns immediately without
    an LLM call — hallucination is structurally impossible.
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def analyze(
        self,
        query: str,
        result_summary: str,
        schema: dict,
        stats_summary: str,
        chat_summary: str,
        mode: str = "executor",
    ) -> dict:
        """
        Returns dict with keys: insight, anomalies, follow_ups, confidence.
        Falls back to safe defaults if LLM fails.
        """
        # ------------------------------------------------------------------
        # Guard: if no real computation happened, return immediately without
        # calling the LLM. Exception: concept mode uses its own sentinel and
        # is handled by the prompt instructions (rule 10).
        # ------------------------------------------------------------------
        if (
            not result_summary
            or result_summary.strip().startswith(_NO_COMPUTE_SENTINEL)
        ):
            return {
                "insight": "Insufficient computed data to draw conclusions.",
                "anomalies": [],
                "follow_ups": [],
                "confidence": "low",
            }
        schema_text = self._format_schema(schema)

        # Truncate inputs aggressively to control token count
        result_truncated = result_summary[:1500] if result_summary else "No result produced."
        stats_truncated = stats_summary[:600] if stats_summary else "No stats available."
        chat_truncated = chat_summary[:200] if chat_summary else "No prior analysis."

        user_content = f"""User Query: {query}

Computed Result:
{result_truncated}

Dataset Schema:
{schema_text}

Dataset Statistical Context:
{stats_truncated}

Conversation History Summary: {chat_truncated}

Analysis Mode: {mode.upper()}

Provide your JSON analysis now."""

        messages = [
            {"role": "system", "content": COGNITIVE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            raw = await self.llm_client.complete(
                messages, temperature=0.3, max_tokens=700
            )
            return self._parse_response(raw)
        except Exception as e:
            logger.warning("CognitiveEngine LLM call failed: %s", e)
            return self._fallback_response()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> dict:
        # Try to extract JSON block (handle markdown code fences if present)
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            try:
                data = json.loads(json_match.group())
                anomalies = data.get("anomalies", [])
                # Enforce max 2 anomalies, validate structure
                cleaned_anomalies = []
                for a in anomalies[:2]:
                    if isinstance(a, dict) and "description" in a:
                        cleaned_anomalies.append({
                            "description": str(a.get("description", "")),
                            "possible_explanation": str(a.get("possible_explanation", "")),
                        })
                follow_ups = data.get("follow_ups", [])
                # Ensure exactly 3, pad or trim
                follow_ups = [str(f) for f in follow_ups[:3]]

                return {
                    "insight": str(data.get("insight", "Analysis complete."))[:600],
                    "anomalies": cleaned_anomalies,
                    "follow_ups": follow_ups,
                    "confidence": data.get("confidence", "medium")
                    if data.get("confidence") in ("high", "medium", "low")
                    else "medium",
                }
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning("CognitiveEngine JSON parse failed: %s", e)

        return self._fallback_response()

    @staticmethod
    def _fallback_response() -> dict:
        return {
            "insight": "Analysis complete. Review the result for details.",
            "anomalies": [],
            "follow_ups": [],
            "confidence": "low",
        }

    @staticmethod
    def _format_schema(schema: dict) -> str:
        if not schema or "columns" not in schema:
            return "Schema not available."
        cols = [f"  - {c['name']} ({c['dtype']})" for c in schema.get("columns", [])]
        return "\n".join(cols[:40])  # cap at 40 columns

    @staticmethod
    def build_stats_summary(summary_stats: dict) -> str:
        """
        Convert the full summary_stats dict into a compact string
        (<600 chars) suitable for the cognitive engine prompt.
        """
        if not summary_stats:
            return "No stats available."
        lines = []
        meta = summary_stats.get("_meta", {})
        if meta:
            lines.append(
                f"Rows: {meta.get('total_rows', '?')}, "
                f"Columns: {meta.get('total_columns', '?')}"
            )
        for col, stats in summary_stats.items():
            if col == "_meta" or not isinstance(stats, dict):
                continue
            parts = []
            for k in ("count", "mean", "min", "max", "std", "top", "freq"):
                v = stats.get(k)
                if v is not None:
                    parts.append(
                        f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                    )
            if parts:
                lines.append(f"  {col}: {', '.join(parts)}")
            if len("\n".join(lines)) > 550:
                lines.append("  ... (truncated)")
                break
        return "\n".join(lines)
