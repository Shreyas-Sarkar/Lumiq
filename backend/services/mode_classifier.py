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
        if col and re.search(r"(?<!\w)" + re.escape(col) + r"(?!\w)", query_clean):
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
        query_clean = query.lower().strip().replace("_", " ")
        columns = [str(c).lower().replace("_", " ") for c in df.columns] if df is not None else []
        dataset_signal = matches_column(query_clean, columns)
        concept = any(re.search(p, query_clean) for p in CONCEPT_PATTERNS)
        if "in general" in query_clean or (concept and not dataset_signal):
            return "concept"
        if any(re.search(p, query_clean) for p in IRRELEVANT_PATTERNS) and not dataset_signal:
            return "irrelevant"
        if any(re.search(p, query_clean) for p in HYBRID_PATTERNS):
            return "hybrid"
        if dataset_signal or any(re.search(p, query_clean) for p in EXECUTOR_PATTERNS):
            return "executor"
        return "concept" if concept else "executor"
