import re

# ---------------------------------------------------------------------------
# Keyword pattern sets
# ---------------------------------------------------------------------------

# Signals a directly computable / retrievable query
EXECUTOR_PATTERNS = [
    r"\b(what is|what are|how many|how much|count|total|average|mean|median)\b",
    r"\b(max|maximum|min|minimum|sum|list|show|top|bottom|filter|group|sort)\b",
    r"\b(rank|highest|lowest|most|least|distribution|percent|percentage|ratio)\b",
    r"\b(find|get|fetch|calculate|compute|give me|tell me the)\b",
    r"\b(unique|distinct|over time|trend|range)\b",
]

# Signals a relationship / comparative query — MUST trigger computation
HYBRID_PATTERNS = [
    r"\b(affect|impact|influence|relationship|correlation|correlated)\b",
    r"\b(compare|between|difference|vs|versus)\b",
    r"\b(does .+ affect|is there a relationship|how does .+ relate)\b",
    r"\b(cause|reason|why does|what drives|predictive)\b",
]

# Signals a purely definitional / conceptual query — NO execution
CONCEPT_PATTERNS = [
    r"\b(what is|what are)\b.{0,60}\b(in general|concept|definition|mean|means)\b",
    r"\b(define|definition of|explain|meaning of|what does .+ mean)\b",
    r"\b(how does .+ work|what is the theory|what is meant by)\b",
]

# Signals clearly off-topic content
IRRELEVANT_PATTERNS = [
    r"\b(weather|temperature|forecast|rain|snow|sunny)\b",
    r"\b(sport|football|cricket|basketball|soccer|match|score)\b",
    r"\b(recipe|cook|bake|ingredient|meal|food)\b",
    r"\b(joke|funny|humor|laugh|meme)\b",
    r"\b(stock market|crypto|bitcoin|nft|blockchain)\b",
    r"\b(personal|my life|my family|my health|my relationship)\b",
]

# ---------------------------------------------------------------------------
# Lightweight alias support — generic only, no dataset-specific words
# ---------------------------------------------------------------------------

COMMON_ALIASES = {
    "amount": ["value", "total"],
    "score":  ["rating", "grade"],
    "count":  ["number", "how many"],
}


# ---------------------------------------------------------------------------
# Schema-aware helpers (pure Python, O(n), zero external deps)
# ---------------------------------------------------------------------------

def matches_column(query_clean: str, normalized_columns: list[str]) -> bool:
    """Return True if any normalized column name appears as a substring in the query."""
    for col in normalized_columns:
        if col and col in query_clean:
            return True
    return False


def matches_alias(query_clean: str) -> bool:
    """Return True if any generic alias for a common concept appears in the query."""
    for _key, aliases in COMMON_ALIASES.items():
        if any(alias in query_clean for alias in aliases):
            return True
    return False


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ModeClassifier:
    """
    Deterministic 4-mode query classifier.

    Output modes:
        executor   — directly computable query; code is generated and executed
        hybrid     — relationship/comparative query; MUST compute metrics
        concept    — definitional query; NO code generated or executed
        irrelevant — off-topic; rejected immediately

    Decision order (strict, top-to-bottom):
        1. Schema-aware column/alias match          → executor
        2. EXECUTOR_PATTERNS keyword match          → executor
        3. HYBRID_PATTERNS keyword match            → hybrid
        4. CONCEPT_PATTERNS keyword match           → concept
        5. IRRELEVANT_PATTERNS keyword match        → irrelevant
        6. Smart default fallback                   → concept | executor

    Architectural invariant:
        concept  → NEVER triggers execution
        hybrid   → ALWAYS triggers execution
        executor → ALWAYS triggers execution
    """

    def classify(self, query: str, df=None) -> str:
        """
        Classify `query` into one of: 'executor', 'hybrid', 'concept', 'irrelevant'.

        Parameters
        ----------
        query : str
            Raw user query.
        df : pandas.DataFrame or None
            Loaded dataset. When provided, column-name matching is applied.
            When None, falls back to keyword-only matching — no crash.
        """
        query_clean = query.lower().strip()

        # ------------------------------------------------------------------
        # 1. Schema-aware matching (only when df is available)
        # ------------------------------------------------------------------
        if df is not None:
            try:
                normalized_columns = [
                    col.lower().replace("_", " ") for col in df.columns.tolist()
                ]
                if matches_column(query_clean, normalized_columns):
                    return "executor"
                if matches_alias(query_clean):
                    return "executor"
            except Exception:
                # Defensive: never let schema matching crash the classifier
                pass

        # ------------------------------------------------------------------
        # 2. Executor keyword patterns
        # ------------------------------------------------------------------
        executor_score = sum(
            bool(re.search(p, query_clean)) for p in EXECUTOR_PATTERNS
        )
        if executor_score > 0:
            return "executor"

        # ------------------------------------------------------------------
        # 3. Hybrid patterns (relationship / comparison) — check BEFORE concept
        #    so "explain correlation between X and Y" → hybrid, not concept
        # ------------------------------------------------------------------
        hybrid_score = sum(
            bool(re.search(p, query_clean)) for p in HYBRID_PATTERNS
        )
        if hybrid_score > 0:
            return "hybrid"

        # ------------------------------------------------------------------
        # 4. Concept patterns (definitional)
        # ------------------------------------------------------------------
        concept_score = sum(
            bool(re.search(p, query_clean)) for p in CONCEPT_PATTERNS
        )
        if concept_score > 0:
            return "concept"

        # ------------------------------------------------------------------
        # 5. Irrelevant patterns
        # ------------------------------------------------------------------
        irrelevant_score = sum(
            bool(re.search(p, query_clean)) for p in IRRELEVANT_PATTERNS
        )
        if irrelevant_score > 0:
            return "irrelevant"

        # ------------------------------------------------------------------
        # 6. Smart default fallback
        #    If the query reads like a question without data signals → concept
        #    Otherwise default to executor (executor-first architecture)
        # ------------------------------------------------------------------
        definitional_signals = re.search(
            r"\b(what is|what are|define|explain|how does|meaning)\b", query_clean
        )
        if definitional_signals:
            return "concept"

        return "executor"
