"use client";

import { useState } from "react";
import {
  ChevronDown,
  ChevronUp,
  Lightbulb,
  AlertTriangle,
  Code2,
  Cpu,
  Brain,
  CheckCircle2,
  MinusCircle,
  XCircle,
} from "lucide-react";
import { Message } from "@/types";
import CodeBlock from "./CodeBlock";
import DataTable from "./DataTable";

interface MessageBubbleProps {
  message: Message;
  onFollowUpClick?: (question: string) => void;
}

// ──────────────────────────────────────────────────────────────────────────────
// Sub-components
// ──────────────────────────────────────────────────────────────────────────────

function ModeBadge({ mode }: { mode: "executor" | "analyst" }) {
  if (mode === "analyst") {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-indigo-500/15 text-indigo-300 border border-indigo-500/25">
        <Brain size={9} />
        Analysis
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-purple-500/15 text-purple-300 border border-purple-500/25">
      <Cpu size={9} />
      Computation
    </span>
  );
}

function ConfidenceDot({ confidence }: { confidence: "high" | "medium" | "low" }) {
  if (confidence === "high") {
    return (
      <span className="inline-flex items-center gap-1 text-[10px] text-emerald-400">
        <CheckCircle2 size={10} />
        High confidence
      </span>
    );
  }
  if (confidence === "medium") {
    return (
      <span className="inline-flex items-center gap-1 text-[10px] text-amber-400">
        <MinusCircle size={10} />
        Medium confidence
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-[10px] text-red-400">
      <XCircle size={10} />
      Low confidence
    </span>
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// Main component
// ──────────────────────────────────────────────────────────────────────────────

export default function MessageBubble({ message, onFollowUpClick }: MessageBubbleProps) {
  const [showCode, setShowCode] = useState(false);
  const [showAnomalies, setShowAnomalies] = useState(false);

  // ── User message ────────────────────────────────────────────────────────────
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="bg-purple-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-[75%]">
          <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">
            {message.content}
          </p>
        </div>
      </div>
    );
  }

  const meta = message.metadata;

  // ── Assistant message (no metadata — plain text e.g. error) ─────────────────
  if (!meta) {
    return (
      <div className="flex items-start gap-3">
        <Avatar />
        <div className="bg-gray-800 text-gray-100 rounded-2xl rounded-tl-sm px-4 py-3 max-w-[85%]">
          <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">
            {message.content}
          </p>
        </div>
      </div>
    );
  }

  const isError = meta.result_type === "error";
  const isRejection = meta.result_type === "rejection";
  const isPlot = meta.result_type === "plot";
  const hasInsight = meta.insight && meta.insight.trim().length > 0;
  const hasAnomalies = meta.anomalies && meta.anomalies.length > 0;
  const hasCode = meta.code && meta.code.trim().length > 0;
  const hasFollowUps = meta.follow_ups && meta.follow_ups.length > 0;
  const plotBase64 =
    isPlot && meta.result && typeof meta.result === "object"
      ? (meta.result as Record<string, unknown>).plot_base64 as string | undefined
      : undefined;

  return (
    <div className="flex items-start gap-3">
      <Avatar />
      <div className="bg-gray-800/80 backdrop-blur text-gray-100 rounded-2xl rounded-tl-sm px-4 py-4 max-w-[87%] w-full border border-gray-700/50 shadow-lg">

        {/* ── Header row: mode badge + confidence ── */}
        {!isError && !isRejection && (
          <div className="flex items-center gap-2 mb-3">
            <ModeBadge mode={meta.mode || "executor"} />
            <ConfidenceDot confidence={meta.confidence || "medium"} />
          </div>
        )}

        {/* ── Error state ── */}
        {isError && (
          <div className="bg-red-900/30 border border-red-700/60 rounded-xl px-4 py-3 mb-3">
            <p className="text-red-300 font-medium text-sm">Analysis Error</p>
            <p className="text-red-400 text-sm mt-1 leading-relaxed">{meta.answer}</p>
            {meta.insight && (
              <p className="text-red-500/80 text-xs mt-2">{meta.insight}</p>
            )}
          </div>
        )}

        {/* ── Rejection state ── */}
        {isRejection && (
          <div className="bg-orange-900/30 border border-orange-700/60 rounded-xl px-4 py-3 mb-3">
            <p className="text-orange-300 font-medium text-sm">Unable to Answer</p>
            <p className="text-orange-400 text-sm mt-1 leading-relaxed">{meta.answer}</p>
            {meta.insight && (
              <p className="text-orange-500/80 text-xs mt-2">{meta.insight}</p>
            )}
          </div>
        )}

        {/* ── Answer ── */}
        {!isError && !isRejection && (
          <p className="text-white text-sm leading-relaxed mb-3 font-medium">
            {meta.answer}
          </p>
        )}

        {/* ── Plot image ── */}
        {isPlot && plotBase64 && (
          <div className="mb-3 rounded-xl overflow-hidden border border-gray-700/50">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`data:image/png;base64,${plotBase64}`}
              alt="Generated visualization"
              className="w-full h-auto"
            />
          </div>
        )}

        {/* ── Data table ── */}
        {meta.result !== null &&
          meta.result !== undefined &&
          !isError &&
          !isRejection &&
          !isPlot && (
            <DataTable
              result={meta.result as Record<string, unknown>}
              resultType={meta.result_type}
            />
          )}

        {/* ── Insight card ── */}
        {hasInsight && !isError && !isRejection && (
          <div className="mt-3 bg-indigo-950/40 border border-indigo-700/30 rounded-xl px-4 py-3">
            <div className="flex items-center gap-1.5 mb-1.5">
              <Lightbulb size={13} className="text-indigo-400 flex-shrink-0" />
              <span className="text-indigo-300 text-xs font-semibold uppercase tracking-wide">
                Insight
              </span>
            </div>
            <p className="text-gray-200 text-sm leading-relaxed">{meta.insight}</p>
          </div>
        )}

        {/* ── Anomalies (collapsible, only if present) ── */}
        {hasAnomalies && (
          <div className="mt-3">
            <button
              onClick={() => setShowAnomalies(!showAnomalies)}
              className="flex items-center gap-1.5 text-xs text-amber-400 hover:text-amber-300 transition-colors"
            >
              <AlertTriangle size={11} />
              {meta.anomalies.length} anomal{meta.anomalies.length === 1 ? "y" : "ies"} detected
              {showAnomalies ? (
                <ChevronUp size={11} />
              ) : (
                <ChevronDown size={11} />
              )}
            </button>
            {showAnomalies && (
              <div className="mt-2 space-y-2">
                {meta.anomalies.map((anomaly, i) => (
                  <div
                    key={i}
                    className="bg-amber-950/30 border border-amber-700/30 rounded-lg px-3 py-2.5"
                  >
                    <p className="text-amber-200 text-xs font-medium">
                      {anomaly.description}
                    </p>
                    {anomaly.possible_explanation && (
                      <p className="text-amber-400/80 text-xs mt-1 leading-relaxed">
                        {anomaly.possible_explanation}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── Code block (collapsed by default) ── */}
        {hasCode && (
          <div className="mt-3 border-t border-gray-700/50 pt-3">
            <button
              onClick={() => setShowCode(!showCode)}
              className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-gray-300 transition-colors"
            >
              <Code2 size={11} />
              {showCode ? "Hide" : "Show"} computation
              {showCode ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
            </button>
            {showCode && (
              <div className="mt-2">
                <CodeBlock code={meta.code} />
              </div>
            )}
          </div>
        )}

        {/* ── Follow-up question pills ── */}
        {hasFollowUps && (
          <div className="mt-4 pt-3 border-t border-gray-700/40">
            <p className="text-gray-500 text-[10px] uppercase tracking-wide mb-2 font-medium">
              Explore further
            </p>
            <div className="flex flex-col gap-1.5">
              {meta.follow_ups.map((q, i) => (
                <button
                  key={i}
                  onClick={() => onFollowUpClick?.(q)}
                  className="text-left text-xs text-gray-300 hover:text-white bg-gray-700/40 hover:bg-gray-700/70 border border-gray-600/40 hover:border-purple-500/40 rounded-lg px-3 py-2 transition-all duration-150 leading-relaxed"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Avatar() {
  return (
    <div className="w-7 h-7 rounded-full bg-gradient-to-br from-purple-600 to-indigo-600 flex items-center justify-center flex-shrink-0 text-xs font-bold mt-0.5 shadow-md">
      L
    </div>
  );
}
