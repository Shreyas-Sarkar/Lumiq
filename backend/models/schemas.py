from pydantic import BaseModel
from typing import Any, Optional
from datetime import datetime


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    nullable: bool


class SchemaInfo(BaseModel):
    columns: list[ColumnInfo]


class DatasetResponse(BaseModel):
    id: str
    chat_id: str
    filename: str
    row_count: int
    column_count: int
    schema_info: dict[str, Any]
    sample_rows: list[dict[str, Any]]
    uploaded_at: str


class UploadResponse(BaseModel):
    dataset: DatasetResponse
    message: str


class ChatCreateRequest(BaseModel):
    title: str = "New Chat"


class ChatResponse(BaseModel):
    id: str
    user_id: str
    title: str
    dataset_id: Optional[str] = None
    status: str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: str
    chat_id: str
    role: str
    content: str
    metadata: Optional[dict[str, Any]] = None
    sequence_number: int
    created_at: str


class QueryRequest(BaseModel):
    chat_id: str
    query: str

    model_config = {"str_max_length": 1000}


class AnomalyItem(BaseModel):
    description: str
    possible_explanation: str


class AssistantResponse(BaseModel):
    """
    Unified response contract for all query modes.

    Fields:
        answer       – Direct 1-2 sentence factual answer to the query.
        insight      – Dataset-specific interpretation of what the result means.
        anomalies    – 0-2 real anomalies surfaced from the result (empty list if none).
        follow_ups   – Exactly 3 suggested next questions (empty list on LLM failure).
        confidence   – "high" | "medium" | "low" based on result completeness.
        code         – Generated Python code (empty string for pure analyst mode).
        result       – Serialized execution output (None for analyst mode).
        result_type  – "dataframe" | "scalar" | "list" | "plot" | "error" | "rejection" | "analyst".
        mode         – "executor" | "analyst".
        rag_context_used – RAG chunks retrieved for this query.
        retry_count  – Number of code generation retries required.
    """
    answer: str
    insight: str
    anomalies: list[AnomalyItem]
    follow_ups: list[str]
    confidence: str
    code: str
    result: Any
    result_type: str
    mode: str
    rag_context_used: list[str]
    retry_count: int


class QueryResponse(BaseModel):
    message_id: str
    response: AssistantResponse
    chat_id: str
    created_at: str


class ExecutionResult(BaseModel):
    success: bool
    output: Any
    output_type: str
    error: Optional[str] = None
    execution_time_ms: float
    serialized_output: str
    plot_base64: Optional[str] = None  # populated when code generates a matplotlib figure


class EvaluationResult(BaseModel):
    is_valid: bool
    is_relevant: bool
    confidence: float
    reason: str
    should_retry: bool
    suggested_fix: Optional[str] = None


class CodeGenerationResult(BaseModel):
    code: str
    raw_llm_response: str
