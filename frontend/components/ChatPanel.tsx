"use client";

import { useRef, useEffect, useState } from "react";
import { Upload, Database, ChevronDown, ChevronUp } from "lucide-react";
import { Message, Dataset, ProcessingPhase } from "@/types";
import AnalysisBlock from "./AnalysisBlock";
import ExecutionLoader from "./ExecutionLoader";
import InputBar from "./InputBar";

interface ChatPanelProps {
  chatId: string;
  messages: Message[];
  dataset: Dataset | null;
  isLoading: boolean;
  processingPhase: ProcessingPhase;
  onSend: (query: string) => void;
  onUpload: () => void;
}

export default function ChatPanel({
  chatId,
  messages,
  dataset,
  isLoading,
  processingPhase,
  onSend,
  onUpload,
}: ChatPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [schemaExpanded, setSchemaExpanded] = useState(false);
  const [prefillQuery, setPrefillQuery] = useState("");

  // Auto-scroll on new messages or loading state change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, processingPhase]);

  function handleFollowUpClick(question: string) {
    setPrefillQuery(question);
  }

  // ── Empty state ──────────────────────────────────────────────────────────
  const isEmpty = messages.length === 0 && !isLoading;

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden" style={{ background: "var(--bg)" }}>

      {/* ── Dataset context bar ── */}
      {dataset && (
        <div
          className="flex-shrink-0 mx-5 mt-4"
          style={{
            border: "1px solid var(--border)",
            borderRadius: "10px",
            background: "var(--surface)",
          }}
        >
          <button
            onClick={() => setSchemaExpanded(!schemaExpanded)}
            className="w-full flex items-center gap-3 px-4 py-3"
          >
            <div
              className="w-6 h-6 rounded flex items-center justify-center flex-shrink-0"
              style={{
                background: "var(--surface-3)",
                border: "1px solid var(--border-strong)",
              }}
            >
              <Database size={11} style={{ color: "var(--muted)" }} />
            </div>
            <div className="flex-1 text-left min-w-0">
              <p className="text-xs font-medium truncate" style={{ color: "var(--text)" }}>
                {dataset.filename}
              </p>
              <p className="text-[10px]" style={{ color: "var(--muted)" }}>
                {dataset.row_count.toLocaleString()} rows · {dataset.column_count} columns
              </p>
            </div>
            <div style={{ color: "var(--muted)" }}>
              {schemaExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            </div>
          </button>

          {schemaExpanded && dataset.schema_info?.columns && (
            <div
              className="px-4 pb-4 pt-1"
              style={{ borderTop: "1px solid var(--border)" }}
            >
              <div className="flex flex-wrap gap-1.5 mt-3">
                {dataset.schema_info.columns.map((col) => (
                  <span
                    key={col.name}
                    title={col.dtype}
                    className="inline-flex items-center gap-1.5 text-[10px] px-2 py-1 rounded font-mono"
                    style={{
                      background: "var(--surface-2)",
                      border: "1px solid var(--border)",
                      color: "var(--muted)",
                    }}
                  >
                    {col.name}
                    <span
                      className="opacity-50"
                      style={{ color: "var(--muted)", fontSize: "9px" }}
                    >
                      {col.dtype}
                    </span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Analysis timeline ── */}
      <div className="flex-1 overflow-y-auto px-5 py-6">
        {isEmpty ? (
          /* ── Empty state ── */
          <div className="flex flex-col items-center justify-center h-full text-center">
            {!dataset ? (
              <div className="max-w-xs">
                <div
                  className="w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-5"
                  style={{
                    background: "var(--surface)",
                    border: "1px solid var(--border-strong)",
                  }}
                >
                  <Upload size={22} style={{ color: "var(--muted)" }} />
                </div>
                <h3 className="text-sm font-semibold mb-2" style={{ color: "var(--text)" }}>
                  No dataset loaded
                </h3>
                <p className="text-xs leading-relaxed mb-5" style={{ color: "var(--muted)" }}>
                  Upload a CSV to start running deterministic analysis against your data.
                </p>
                <button
                  onClick={onUpload}
                  id="upload-dataset-btn"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all duration-150"
                  style={{
                    background: "var(--surface)",
                    border: "1px solid var(--border-strong)",
                    color: "var(--text)",
                  }}
                  onMouseEnter={(e) =>
                    ((e.currentTarget as HTMLButtonElement).style.background =
                      "var(--surface-2)")
                  }
                  onMouseLeave={(e) =>
                    ((e.currentTarget as HTMLButtonElement).style.background =
                      "var(--surface)")
                  }
                >
                  <Upload size={13} />
                  Upload CSV
                </button>
              </div>
            ) : (
              <div className="max-w-xs">
                <div
                  className="w-12 h-12 rounded-xl flex items-center justify-center mx-auto mb-5"
                  style={{
                    background: "var(--surface)",
                    border: "1px solid var(--border-strong)",
                  }}
                >
                  <Database size={22} style={{ color: "var(--muted)" }} />
                </div>
                <h3 className="text-sm font-semibold mb-2" style={{ color: "var(--text)" }}>
                  Ready to analyze
                </h3>
                <p className="text-xs leading-relaxed" style={{ color: "var(--muted)" }}>
                  Ask a question about{" "}
                  <span style={{ color: "var(--text)" }}>{dataset.filename}</span>
                  . Results are computed from real code execution.
                </p>
                <div
                  className="mt-5 text-left rounded-lg px-4 py-3 text-xs"
                  style={{
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                    color: "var(--muted)",
                  }}
                >
                  <p className="mb-1 font-medium" style={{ color: "var(--text)" }}>
                    Try asking:
                  </p>
                  <p>"What is the average of each numeric column?"</p>
                  <p className="mt-0.5">"Show the top 10 rows by revenue."</p>
                  <p className="mt-0.5">"Is there a correlation between age and salary?"</p>
                </div>
              </div>
            )}
          </div>
        ) : (
          /* ── Messages timeline ── */
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg) => (
              <AnalysisBlock
                key={msg.id}
                message={msg}
                onFollowUpClick={handleFollowUpClick}
              />
            ))}

            {/* Execution loader */}
            {isLoading && processingPhase !== "idle" && (
              <ExecutionLoader currentPhase={processingPhase} />
            )}

            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* ── Input bar ── */}
      <InputBar
        onSend={onSend}
        onUpload={onUpload}
        isLoading={isLoading}
        hasDataset={!!dataset}
        prefillQuery={prefillQuery}
        onPrefillConsumed={() => setPrefillQuery("")}
      />
    </div>
  );
}
