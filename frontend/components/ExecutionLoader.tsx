"use client";

import { ProcessingPhase, PHASE_LABELS } from "@/types";
import { Check } from "lucide-react";

interface ExecutionLoaderProps {
  currentPhase: ProcessingPhase;
}

const PHASE_ORDER: ProcessingPhase[] = [
  "classifying",
  "retrieving",
  "generating",
  "executing",
  "evaluating",
  "enriching",
];

export default function ExecutionLoader({ currentPhase }: ExecutionLoaderProps) {
  const currentIndex = PHASE_ORDER.indexOf(currentPhase);

  return (
    <div
      className="rounded-xl overflow-hidden animate-fade-slide-in"
      style={{
        border: "1px solid var(--border)",
        background: "var(--surface)",
        opacity: 0,
      }}
    >
      {/* Header */}
      <div
        className="flex items-center gap-3 px-5 py-3 border-b"
        style={{ borderColor: "var(--border)", background: "var(--surface-2)" }}
      >
        <div
          className="w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold"
          style={{
            background: "var(--surface-3)",
            border: "1px solid var(--border-strong)",
            color: "var(--text)",
          }}
        >
          L
        </div>
        <span className="text-xs font-semibold" style={{ color: "var(--text)" }}>
          Analysis in progress
        </span>
        <div className="ml-auto flex items-center gap-1.5">
          <span
            className="w-1.5 h-1.5 rounded-full animate-pulse-dot"
            style={{ background: "#22c55e" }}
          />
          <span className="text-[10px]" style={{ color: "var(--muted)" }}>
            Running
          </span>
        </div>
      </div>

      {/* Phase steps */}
      <div className="px-5 py-4 space-y-2.5">
        {PHASE_ORDER.slice(0, currentIndex + 1).map((phase, i) => {
          const isDone    = i < currentIndex;
          const isCurrent = i === currentIndex;
          return (
            <div key={phase} className="flex items-center gap-3">
              {/* Icon */}
              <div
                className="w-5 h-5 rounded flex items-center justify-center flex-shrink-0"
                style={{
                  border: `1px solid ${
                    isDone
                      ? "rgba(34, 197, 94, 0.25)"
                      : isCurrent
                      ? "var(--border-strong)"
                      : "var(--border)"
                  }`,
                  background: isDone
                    ? "rgba(34, 197, 94, 0.08)"
                    : "var(--surface-3)",
                }}
              >
                {isDone ? (
                  <Check size={10} style={{ color: "#22c55e" }} />
                ) : isCurrent ? (
                  <span
                    className="w-1.5 h-1.5 rounded-full animate-pulse-dot"
                    style={{ background: "var(--accent)" }}
                  />
                ) : null}
              </div>

              {/* Label */}
              <span
                className="text-xs font-medium"
                style={{
                  color: isDone
                    ? "var(--muted)"
                    : isCurrent
                    ? "var(--text)"
                    : "var(--muted)",
                  opacity: isDone ? 0.6 : 1,
                }}
              >
                {PHASE_LABELS[phase]}
              </span>

              {/* Skeleton shimmer for current */}
              {isCurrent && (
                <div
                  className="ml-auto skeleton rounded"
                  style={{ width: "60px", height: "8px" }}
                />
              )}
            </div>
          );
        })}

        {/* Skeleton preview rows for upcoming phases */}
        {currentIndex < PHASE_ORDER.length - 1 && (
          <div className="mt-4 space-y-2">
            <div className="skeleton rounded" style={{ width: "100%", height: "36px" }} />
            <div className="skeleton rounded" style={{ width: "75%", height: "10px" }} />
          </div>
        )}
      </div>
    </div>
  );
}
