export interface DatasetProfile {
  version: number;
  rows: number;
  duplicate_rows: number;
  memory_bytes: number;
  columns: {
    name: string;
    dtype: string;
    kind: string;
    missing: number;
    unique: number;
    start?: string;
    end?: string;
    invalid_dates?: number;
    examples?: string[];
  }[];
  limitations: string[];
}
export interface AnalysisStep {
  operation: "aggregate" | "group_by" | "compare_periods" | "correlation";
  metric: string;
  aggregation?: string;
  group_by?: string | null;
  date_column?: string | null;
  periods?: string[][];
  filters?: { column: string; operator: string; value: unknown }[];
  limit?: number;
  sort_descending?: boolean;
}
export interface AnalysisPlan {
  version?: number;
  title?: string;
  explanation?: string;
  assumptions?: string[];
  clarification?: string | null;
  concept_answer?: string | null;
  steps: AnalysisStep[];
}
export interface Evidence {
  id: string;
  operation: string;
  metric: string;
  aggregation: string;
  summary: Record<string, number | null>;
  rows: Record<string, unknown>[];
  chart: {
    kind: string;
    x: string;
    y: string;
    rows: Record<string, unknown>[];
  } | null;
  limitations: string[];
  input_rows: number;
  filtered_rows: number;
  hash: string;
  filters: { column: string; operator: string; value: unknown }[];
  periods: string[][];
}
export interface Investigation {
  id: string;
  chat_id: string;
  investigation_id: string;
  created_at: string;
  revision: number;
  state: {
    status: string;
    question: string;
    plan?: AnalysisPlan;
    profile?: DatasetProfile;
    evidence: Evidence[];
    findings: {
      evidence_id: string;
      observation: string;
      hypothesis?: string | null;
    }[];
    concept_answer?: string;
    follow_ups?: string[];
    http_attempts: number;
    provider_calls: {
      stage: string;
      provider: string;
      model: string;
      outcome: string;
    }[];
    events: { status: string; at: string; detail?: string }[];
    error?: {
      category: string;
      message: string;
      resumable: boolean;
      retry_at?: string;
      reasons?: { provider: string; reason: string }[];
    };
  };
}
