"use client";

import { useState } from "react";
import {
  ChevronDown,
  ChevronUp,
  Lightbulb,
  AlertTriangle,
  Code2,
  Terminal,
  Brain,
  CheckCircle2,
  MinusCircle,
  XCircle,
  Copy,
  Check,
  ArrowRight,
} from "lucide-react";
import { Message, AssistantMetadata } from "@/types";
import DataTable from "./DataTable";
import SyntaxHighlighter from "react-syntax-highlighter";
import { atomOneDark } from "react-syntax-highlighter/dist/esm/styles/hljs";

// ─── Types ────────────────────────────────────────────────────────────────────

interface AnalysisBlockProps {
  message: Message;
  onFollowUpClick?: (question: string) => void;
}

// ─── Confidence pill ──────────────────────────────────────────────────────────

function ConfidencePill({ confidence }: { confidence: "high" | "medium" | "low" }) {
  const map = {
    high: { icon: CheckCircle2, color: "#22c55e", label: "High confidence" },
    medium: { icon: MinusCircle, color: "#f59e0b", label: "Medium confidence" },
    low: { icon: XCircle, color: "#ef4444", label: "Low confidence" },
  };
  const { icon: Icon, color, label } = map[confidence] ?? map.medium;
  return (
    <span
      className="inline-flex items-center gap-1 text-[10px] font-medium"
      style={{ color }}
    >
      <Icon size={10} />
      {label}
    </span>
  );
}

// ─── Mode badge ───────────────────────────────────────────────────────────────

function ModeBadge({ mode }: { mode: string }) {
  const isAnalyst = mode === "analyst" || mode === "concept";
  return (
    <span
      className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded"
      style={{
        background: "var(--surface-3)",
        border: "1px solid var(--border-strong)",
        color: "var(--muted)",
      }}
    >
      {isAnalyst ? <Brain size={9} /> : <Terminal size={9} />}
      {mode}
    </span>
  );
}

// ─── Code panel ───────────────────────────────────────────────────────────────

function CodePanel({ code }: { code: string }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!code) return null;

  async function handleCopy() {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div
      className="rounded-lg overflow-hidden"
      style={{ border: "1px solid var(--border)" }}
    >
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-2.5 transition-colors"
        style={{ background: "var(--surface-3)" }}
        onMouseEnter={(e) =>
          ((e.currentTarget as HTMLButtonElement).style.background = "var(--surface-2)")
        }
        onMouseLeave={(e) =>
          ((e.currentTarget as HTMLButtonElement).style.background = "var(--surface-3)")
        }
      >
        <span className="flex items-center gap-2 text-xs font-medium" style={{ color: "var(--muted)" }}>
          <Code2 size={11} />
          View computation
        </span>
        {open ? (
          <ChevronUp size={11} style={{ color: "var(--muted)" }} />
        ) : (
          <ChevronDown size={11} style={{ color: "var(--muted)" }} />
        )}
      </button>

      {open && (
        <div className="animate-fade-in">
          <div
            className="flex items-center justify-between px-4 py-2 border-t border-b"
            style={{
              background: "#0d0d0d",
              borderColor: "var(--border)",
            }}
          >
            <span className="text-[10px] font-mono" style={{ color: "var(--muted)" }}>
              python
            </span>
            <button
              onClick={handleCopy}
              className="flex items-center gap-1 text-[10px] transition-colors"
              style={{ color: "var(--muted)" }}
              onMouseEnter={(e) =>
                ((e.currentTarget as HTMLButtonElement).style.color = "var(--text)")
              }
              onMouseLeave={(e) =>
                ((e.currentTarget as HTMLButtonElement).style.color = "var(--muted)")
              }
            >
              {copied ? (
                <>
                  <Check size={10} /> Copied
                </>
              ) : (
                <>
                  <Copy size={10} /> Copy
                </>
              )}
            </button>
          </div>
          <div className="max-h-72 overflow-y-auto" style={{ background: "#0d0d0d" }}>
            <SyntaxHighlighter
              language="python"
              style={atomOneDark}
              customStyle={{
                margin: 0,
                padding: "1rem",
                background: "transparent",
                fontSize: "0.75rem",
                lineHeight: "1.6",
              }}
            >
              {code}
            </SyntaxHighlighter>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Anomalies panel ──────────────────────────────────────────────────────────

function AnomaliesPanel({
  anomalies,
}: {
  anomalies: { description: string; possible_explanation: string }[];
}) {
  const [open, setOpen] = useState(false);
  if (!anomalies || anomalies.length === 0) return null;

  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-xs transition-colors"
        style={{ color: "#f59e0b" }}
        onMouseEnter={(e) =>
          ((e.currentTarget as HTMLButtonElement).style.opacity = "0.8")
        }
        onMouseLeave={(e) =>
          ((e.currentTarget as HTMLButtonElement).style.opacity = "1")
        }
      >
        <AlertTriangle size={11} />
        {anomalies.length} anomal{anomalies.length === 1 ? "y" : "ies"} detected
        {open ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
      </button>

      {open && (
        <div className="mt-3 space-y-2 animate-fade-in">
          {anomalies.map((a, i) => (
            <div
              key={i}
              className="rounded-lg px-4 py-3"
              style={{
                background: "rgba(245, 158, 11, 0.05)",
                border: "1px solid rgba(245, 158, 11, 0.15)",
              }}
            >
              <p className="text-xs font-medium mb-0.5" style={{ color: "#fbbf24" }}>
                {a.description}
              </p>
              {a.possible_explanation && (
                <p className="text-xs leading-relaxed" style={{ color: "rgba(251,191,36,0.6)" }}>
                  {a.possible_explanation}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Main AnalysisBlock ───────────────────────────────────────────────────────

export default function AnalysisBlock({ message, onFollowUpClick }: AnalysisBlockProps) {
  // ── User query row ──────────────────────────────────────────────────────────
  if (message.role === "user") {
    return (
      <div className="animate-fade-slide-in" style={{ opacity: 0 }}>
        {/* Query label */}
        <div className="flex items-center gap-2 mb-2">
          <div
            className="h-px flex-1"
            style={{ background: "var(--border)" }}
          />
          <span
            className="text-[10px] font-medium uppercase tracking-widest px-2"
            style={{ color: "var(--muted)" }}
          >
            Query
          </span>
          <div
            className="h-px flex-1"
            style={{ background: "var(--border)" }}
          />
        </div>
        {/* Query bubble */}
        <div className="flex justify-end">
          <div
            className="max-w-[80%] rounded-xl rounded-tr-sm px-4 py-3"
            style={{
              background: "var(--surface-2)",
              border: "1px solid var(--border-strong)",
            }}
          >
            <p className="text-sm leading-relaxed whitespace-pre-wrap break-words"
               style={{ color: "var(--text)" }}>
              {message.content}
            </p>
          </div>
        </div>
      </div>
    );
  }

  // ── Error: no metadata ──────────────────────────────────────────────────────
  if (!message.metadata) {
    return (
      <div
        className="rounded-xl px-5 py-4 animate-fade-slide-in"
        style={{
          background: "rgba(239, 68, 68, 0.05)",
          border: "1px solid rgba(239, 68, 68, 0.2)",
          opacity: 0,
        }}
      >
        <p className="text-sm" style={{ color: "#fca5a5" }}>
          {message.content}
        </p>
      </div>
    );
  }

  const meta = message.metadata as AssistantMetadata;

  const isError      = meta.result_type === "error";
  const isRejection  = meta.result_type === "rejection";
  const isPlot       = meta.result_type === "plot";
  const hasInsight   = meta.insight && meta.insight.trim().length > 0;
  const hasAnomalies = meta.anomalies && meta.anomalies.length > 0;
  const hasCode      = meta.code && meta.code.trim().length > 0;
  const hasFollowUps = meta.follow_ups && meta.follow_ups.length > 0;
  const hasResult    = meta.result !== null && meta.result !== undefined;

  const plotBase64 =
    isPlot && meta.result && typeof meta.result === "object"
      ? (meta.result as Record<string, unknown>).plot_base64 as string | undefined
      : undefined;

  // ── Error block ─────────────────────────────────────────────────────────────
  if (isError) {
    return (
      <div
        className="rounded-xl overflow-hidden animate-fade-slide-in"
        style={{
          border: "1px solid rgba(239, 68, 68, 0.2)",
          background: "var(--surface)",
          opacity: 0,
        }}
      >
        <div
          className="flex items-center gap-2 px-5 py-3 border-b"
          style={{
            background: "rgba(239, 68, 68, 0.06)",
            borderColor: "rgba(239, 68, 68, 0.2)",
          }}
        >
          <XCircle size={13} style={{ color: "#ef4444" }} />
          <span className="text-xs font-semibold" style={{ color: "#fca5a5" }}>
            Execution failed
          </span>
          {meta.retry_count > 0 && (
            <span className="text-[10px] ml-auto" style={{ color: "var(--muted)" }}>
              {meta.retry_count} retr{meta.retry_count === 1 ? "y" : "ies"}
            </span>
          )}
        </div>
        <div className="px-5 py-4 space-y-3">
          <p className="text-sm leading-relaxed" style={{ color: "#fca5a5" }}>
            {meta.answer}
          </p>
          {meta.insight && (
            <p className="text-xs leading-relaxed" style={{ color: "var(--muted)" }}>
              {meta.insight}
            </p>
          )}
          {hasCode && <CodePanel code={meta.code} />}
        </div>
      </div>
    );
  }

  // ── Rejection block ─────────────────────────────────────────────────────────
  if (isRejection) {
    return (
      <div
        className="rounded-xl overflow-hidden animate-fade-slide-in"
        style={{
          border: "1px solid rgba(245, 158, 11, 0.2)",
          background: "var(--surface)",
          opacity: 0,
        }}
      >
        <div
          className="flex items-center gap-2 px-5 py-3 border-b"
          style={{
            background: "rgba(245, 158, 11, 0.06)",
            borderColor: "rgba(245, 158, 11, 0.2)",
          }}
        >
          <AlertTriangle size={13} style={{ color: "#f59e0b" }} />
          <span className="text-xs font-semibold" style={{ color: "#fbbf24" }}>
            Unable to answer
          </span>
        </div>
        <div className="px-5 py-4">
          <p className="text-sm leading-relaxed" style={{ color: "#fbbf24" }}>
            {meta.answer}
          </p>
          {meta.insight && (
            <p className="text-xs leading-relaxed mt-2" style={{ color: "var(--muted)" }}>
              {meta.insight}
            </p>
          )}
        </div>
      </div>
    );
  }

  // ── Standard analysis block ─────────────────────────────────────────────────
  return (
    <div
      className="rounded-xl overflow-hidden animate-fade-slide-in"
      style={{
        border: "1px solid var(--border)",
        background: "var(--surface)",
        opacity: 0,
      }}
    >
      {/* ── Header ── */}
      <div
        className="flex items-center gap-3 px-5 py-3 border-b"
        style={{ borderColor: "var(--border)", background: "var(--surface-2)" }}
      >
        <div
          className="w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold flex-shrink-0"
          style={{
            background: "var(--surface-3)",
            border: "1px solid var(--border-strong)",
            color: "var(--text)",
          }}
        >
          L
        </div>
        <span className="text-xs font-semibold" style={{ color: "var(--text)" }}>
          Analysis
        </span>
        <div className="ml-auto flex items-center gap-3">
          <ModeBadge mode={meta.mode || "executor"} />
          <ConfidencePill confidence={meta.confidence || "medium"} />
        </div>
      </div>

      <div className="px-5 py-5 space-y-5">
        {/* ── Answer ── */}
        <div>
          <p className="text-sm leading-relaxed font-medium" style={{ color: "var(--text)" }}>
            {meta.answer}
          </p>
        </div>

        {/* ── Primary result: scalar ── */}
        {hasResult && !isPlot && meta.result_type === "scalar" && (
          <div
            className="rounded-lg px-5 py-4 flex items-center justify-center"
            style={{ border: "1px solid var(--border)", background: "var(--surface-3)" }}
          >
            <span
              className="text-3xl font-semibold font-mono tracking-tight"
              style={{ color: "var(--text)" }}
            >
              {String(meta.result)}
            </span>
          </div>
        )}

        {/* ── Primary result: table ── */}
        {hasResult && !isError && !isRejection && !isPlot && meta.result_type !== "scalar" && (
          <DataTable
            result={meta.result as Record<string, unknown>}
            resultType={meta.result_type}
          />
        )}

        {/* ── Primary result: plot ── */}
        {isPlot && plotBase64 && (
          <div
            className="rounded-lg overflow-hidden"
            style={{ border: "1px solid var(--border)" }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/png;base64,${plotBase64}`}
              alt="Generated visualization"
              className="w-full h-auto"
            />
          </div>
        )}

        {/* ── Insight ── */}
        {hasInsight && (
          <div
            className="rounded-lg px-4 py-4"
            style={{
              border: "1px solid var(--border)",
              background: "var(--surface-3)",
            }}
          >
            <div className="flex items-center gap-1.5 mb-2">
              <Lightbulb size={12} style={{ color: "var(--accent)" }} />
              <span
                className="text-[10px] font-semibold uppercase tracking-widest"
                style={{ color: "var(--muted)" }}
              >
                Insight
              </span>
            </div>
            <p className="text-sm leading-relaxed" style={{ color: "var(--text)" }}>
              {meta.insight}
            </p>
          </div>
        )}

        {/* ── Anomalies ── */}
        {hasAnomalies && <AnomaliesPanel anomalies={meta.anomalies} />}

        {/* ── Code ── */}
        {hasCode && (
          <CodePanel code={meta.code} />
        )}

        {/* ── Follow-ups ── */}
        {hasFollowUps && (
          <div
            className="pt-4 border-t"
            style={{ borderColor: "var(--border)" }}
          >
            <p
              className="text-[10px] uppercase tracking-widest mb-3 font-medium"
              style={{ color: "var(--muted)" }}
            >
              Continue exploring
            </p>
            <div className="space-y-2">
              {meta.follow_ups.map((q, i) => (
                <button
                  key={i}
                  onClick={() => onFollowUpClick?.(q)}
                  className="w-full text-left rounded-lg px-4 py-2.5 flex items-center gap-3 transition-all duration-150 group"
                  style={{
                    background: "var(--surface-2)",
                    border: "1px solid var(--border)",
                  }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--border-strong)";
                    (e.currentTarget as HTMLButtonElement).style.background = "var(--surface-3)";
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--border)";
                    (e.currentTarget as HTMLButtonElement).style.background = "var(--surface-2)";
                  }}
                >
                  <ArrowRight
                    size={11}
                    className="flex-shrink-0 opacity-40 group-hover:opacity-100 transition-opacity"
                    style={{ color: "var(--muted)" }}
                  />
                  <span className="text-xs leading-relaxed" style={{ color: "var(--text)" }}>
                    {q}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
