export interface User {
  id: string;
  email: string;
  created_at: string;
}

export interface Chat {
  id: string;
  user_id: string;
  title: string;
  dataset_id: string | null;
  status: "active" | "archived";
  created_at: string;
  updated_at: string;
}

export interface AnomalyItem {
  description: string;
  possible_explanation: string;
}

export interface AssistantMetadata {
  answer: string;
  insight: string;
  anomalies: AnomalyItem[];
  follow_ups: string[];
  confidence: "high" | "medium" | "low";
  code: string;
  result: unknown;
  result_type: "dataframe" | "scalar" | "list" | "plot" | "error" | "rejection" | "analyst";
  mode: "executor" | "analyst";
  rag_context_used: string[];
  retry_count: number;
}

export interface Message {
  id: string;
  chat_id: string;
  role: "user" | "assistant";
  content: string;
  metadata: AssistantMetadata | null;
  sequence_number: number;
  created_at: string;
}

export interface Dataset {
  id: string;
  chat_id: string;
  filename: string;
  row_count: number;
  column_count: number;
  schema_info: { columns: ColumnInfo[] };
  sample_rows: Record<string, unknown>[];
  uploaded_at: string;
}

export interface ColumnInfo {
  name: string;
  dtype: string;
  nullable: boolean;
}

export type ProcessingPhase =
  | "classifying"
  | "retrieving"
  | "generating"
  | "executing"
  | "evaluating"
  | "enriching"
  | "idle";

export const PHASE_LABELS: Record<ProcessingPhase, string> = {
  classifying: "Analyzing your question…",
  retrieving: "Retrieving dataset context…",
  generating: "Generating analysis code…",
  executing: "Running computation…",
  evaluating: "Validating result…",
  enriching: "Generating insights…",
  idle: "",
};
