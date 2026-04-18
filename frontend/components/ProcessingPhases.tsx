"use client";

import { ProcessingPhase, PHASE_LABELS } from "@/types";

interface ProcessingPhasesProps {
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

export default function ProcessingPhases({
  currentPhase,
}: ProcessingPhasesProps) {
  const currentIndex = PHASE_ORDER.indexOf(currentPhase);

  return (
    <div className="flex items-start gap-3 max-w-[85%]">
      <div className="w-7 h-7 rounded-full bg-gradient-to-br from-purple-600 to-indigo-600 flex items-center justify-center flex-shrink-0 text-xs font-bold mt-0.5 shadow-md">
        L
      </div>
      <div className="bg-gray-800/80 backdrop-blur border border-gray-700/50 shadow-lg text-gray-100 rounded-2xl rounded-tl-sm px-4 py-4">
        <div className="space-y-2">
          {PHASE_ORDER.slice(0, currentIndex + 1).map((phase, i) => {
            const isDone = i < currentIndex;
            const isCurrent = i === currentIndex;
            return (
              <div key={phase} className="flex items-center gap-2.5 text-sm">
                {isDone ? (
                  <span className="text-gray-500 text-xs w-3 text-center">✓</span>
                ) : isCurrent ? (
                  <span className="w-2 h-2 rounded-full bg-purple-400 animate-pulse ml-0.5 inline-block" />
                ) : (
                  <span className="w-3 text-center" />
                )}
                <span
                  className={
                    isDone
                      ? "text-gray-500 font-medium"
                      : isCurrent
                      ? "text-white font-medium drop-shadow-sm"
                      : "text-gray-400"
                  }
                >
                  {PHASE_LABELS[phase]}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
