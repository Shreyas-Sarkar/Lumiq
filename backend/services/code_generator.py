import re
from models.schemas import CodeGenerationResult

# System prompt for EXECUTOR mode
EXECUTOR_SYSTEM_PROMPT = """You are a Python data analysis expert. Your ONLY job is to write Pandas code to answer questions about a given dataset.

RULES:
1. Return ONLY a Python code block. No explanation before or after.
2. The DataFrame is already loaded as variable `df`. Do not load from files.
3. The following are already available — do NOT import them:
   - `pd`  (pandas)
   - `np`  (numpy)
   - `plt` (matplotlib.pyplot — Agg backend, non-interactive)
   - `sns` (seaborn)
   - `math`, `datetime`, `statistics`, `re` are also available.
4. Store the final result in a variable named `result`.
5. `result` must be a DataFrame, scalar (int/float/str), or list.
6. For visualizations: call plt.figure() to start the plot, build it with plt/sns calls,
   then call plt.show() to finalize. Set `result = None` for plot-only queries.
   The engine captures figures automatically via plt.get_fignums().
7. NEVER use: os, sys, subprocess, open(), exec(), eval(), requests, socket, or any file I/O.
8. NEVER modify the original `df` in place for operations that require the original later.
9. Handle NaN values explicitly where appropriate.
10. If the question cannot be answered with the available columns, raise a ValueError with a clear message.
11. Keep code concise and correct. No comments needed.

Format your response EXACTLY as:
```python
[your code here]
```"""

# System prompt for HYBRID mode — computes statistical evidence for relationship/comparison queries
HYBRID_SYSTEM_PROMPT = """You are a Python data analysis expert. Your task is to generate code that computes ALL REQUIRED STATISTICAL EVIDENCE to answer relationship and comparison questions.

STRICT COMPUTATION RULES:
1. Return ONLY a Python code block. No explanation before or after.
2. The DataFrame is already loaded as variable `df`. Do not load from files.
3. The following are already available — do NOT import them:
   - `pd`  (pandas)
   - `np`  (numpy)
   - `plt` (matplotlib.pyplot — Agg backend, non-interactive)
   - `sns` (seaborn)
   - `math`, `datetime`, `statistics`, `re` are also available.
4. Store the final result in a variable named `result`.
5. REQUIRED PATTERNS — apply based on query type:
   - "relationship", "affect", "impact": MUST compute df[col1].corr(df[col2]) or full correlation matrix.
   - "compare", "between", "vs": MUST use df.groupby(col)[metric].agg([...]).
   - "distribution": MUST use df[col].value_counts() or percentiles.
   - "plot", "graph", "visualize", "chart": MUST call plt.figure(), build the plot with plt/sns calls,
     then call plt.show() to finalize. Set result = None for plot-only queries.
6. MUST compute ALL required metrics — partial results are forbidden.
7. MUST NOT infer or assert relationships without computing them first.
8. `result` must be a DataFrame, scalar, or list containing the full computed evidence.
9. NEVER use: os, sys, subprocess, open(), exec(), eval(), requests, socket, or any file I/O.
10. Handle NaN values explicitly (use dropna() or fillna() where appropriate).
11. If the required columns do not exist in `df`, raise a ValueError with a clear message naming the missing columns.

Format your response EXACTLY as:
```python
[your code here]
```"""

FORBIDDEN_PATTERNS = [
    r"\bimport\s+os\b",
    r"\bimport\s+sys\b",
    r"\bsubprocess\b",
    r"\bopen\s*\(",
    r"\brequests\b",
    r"\bsocket\b",
    r"\b__import__\b",
    r"\.write\s*\(",
    r"\.to_csv\s*\(",
    r"\.to_excel\s*\(",
    r"\bexec\s*\(",
    r"\beval\s*\(",
]


class CodeGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def generate_code(
        self,
        query: str,
        schema_context: list[str],
        chat_history: list[dict],
        dataset_schema: dict,
        retry_suffix: str = "",
        mode: str = "executor",
    ) -> CodeGenerationResult:
        formatted_schema = self._format_schema(dataset_schema)
        rag_context_joined = (
            "\n\n".join(schema_context) if schema_context else "No context retrieved."
        )
        formatted_history = self._format_history(chat_history)

        system_prompt = (
            HYBRID_SYSTEM_PROMPT if mode == "hybrid" else EXECUTOR_SYSTEM_PROMPT
        )

        user_prompt = f"""Dataset Schema:
{formatted_schema}

Retrieved Context:
{rag_context_joined}

Recent Conversation:
{formatted_history}

User Question: {query}

Write Python/Pandas code to answer this question. The DataFrame is loaded as `df`.{retry_suffix}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        raw_response = await self.llm_client.complete_with_retry(messages)
        code = self.extract_code_from_response(raw_response)
        return CodeGenerationResult(code=code, raw_llm_response=raw_response)

    def extract_code_from_response(self, response: str) -> str:
        pattern = r"```python\s*([\s\S]*?)```"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        pattern2 = r"```\s*([\s\S]*?)```"
        match2 = re.search(pattern2, response)
        if match2:
            return match2.group(1).strip()
        return response.strip()

    def validate_code_safety(self, code: str) -> tuple[bool, str]:
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Forbidden pattern detected: {pattern}"
        return True, ""

    def _format_schema(self, schema: dict) -> str:
        if not schema or "columns" not in schema:
            return "No schema available"
        lines = []
        for col in schema["columns"]:
            line = f"  - {col['name']} ({col['dtype']})"
            if col.get("nullable"):
                line += " [nullable]"
            lines.append(line)
        return "\n".join(lines)

    def _format_history(self, history: list[dict]) -> str:
        if not history:
            return "No previous conversation."
        lines = []
        for msg in history[-6:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            lines.append(f"{role.capitalize()}: {content}")
        return "\n".join(lines)
