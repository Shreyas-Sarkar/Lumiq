"use client";

import { useState, useEffect, useRef } from "react";
import { Send, Paperclip, Loader2 } from "lucide-react";

interface InputBarProps {
  onSend: (query: string) => void;
  onUpload: () => void;
  isLoading: boolean;
  hasDataset: boolean;
  prefillQuery?: string;
  onPrefillConsumed?: () => void;
}

export default function InputBar({
  onSend,
  onUpload,
  isLoading,
  hasDataset,
  prefillQuery,
  onPrefillConsumed,
}: InputBarProps) {
  const [query, setQuery] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const MAX_CHARS = 1000;

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        120
      )}px`;
    }
  }, [query]);

  // Follow-up prefill
  useEffect(() => {
    if (prefillQuery) {
      setQuery(prefillQuery);
      textareaRef.current?.focus();
      onPrefillConsumed?.();
    }
  }, [prefillQuery, onPrefillConsumed]);

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleSend() {
    const trimmed = query.trim();
    if (!trimmed || isLoading || trimmed.length > MAX_CHARS) return;
    onSend(trimmed);
    setQuery("");
  }

  const isOverLimit = query.length > MAX_CHARS;
  const canSend = query.trim().length > 0 && !isLoading && !isOverLimit;

  return (
    <div
      className="px-4 py-4 flex-shrink-0"
      style={{
        background: "var(--surface)",
        borderTop: "1px solid var(--border)",
      }}
    >
      <div className="max-w-3xl mx-auto">
        <div
          className="flex items-end gap-2 rounded-xl px-3 py-2.5 transition-all duration-150"
          style={{
            background: "var(--surface-2)",
            border: "1px solid var(--border-strong)",
          }}
          onFocusCapture={(e) => {
            (e.currentTarget as HTMLDivElement).style.borderColor = "rgba(255,255,255,0.18)";
          }}
          onBlurCapture={(e) => {
            (e.currentTarget as HTMLDivElement).style.borderColor = "var(--border-strong)";
          }}
        >
          {/* Upload button */}
          <button
            onClick={onUpload}
            disabled={isLoading}
            title={hasDataset ? "Replace dataset" : "Upload CSV"}
            className="flex-shrink-0 rounded-lg p-1.5 transition-colors disabled:opacity-40"
            style={{ color: "var(--muted)" }}
            onMouseEnter={(e) =>
              ((e.currentTarget as HTMLButtonElement).style.color = "var(--text)")
            }
            onMouseLeave={(e) =>
              ((e.currentTarget as HTMLButtonElement).style.color = "var(--muted)")
            }
          >
            <Paperclip size={16} />
          </button>

          {/* Textarea */}
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              rows={1}
              placeholder={
                hasDataset
                  ? "Ask a question about your data…"
                  : "Upload a CSV first, then ask questions…"
              }
              className="w-full bg-transparent border-none resize-none text-sm leading-relaxed"
              style={{
                color: "var(--text)",
                outline: "none",
                fontFamily: "inherit",
                lineHeight: "1.6",
                opacity: isLoading ? 0.5 : 1,
                cursor: isLoading ? "not-allowed" : "text",
                background: "transparent",
                border: "none",
                resize: "none",
              }}
            />
            {query.length > 800 && (
              <span
                className="absolute bottom-0 right-0 text-[10px]"
                style={{ color: isOverLimit ? "#ef4444" : "var(--muted)" }}
              >
                {query.length}/{MAX_CHARS}
              </span>
            )}
          </div>

          {/* Send button */}
          <button
            onClick={handleSend}
            disabled={!canSend}
            className="flex-shrink-0 rounded-lg p-1.5 transition-all duration-150"
            style={{
              background: canSend ? "var(--surface-3)" : "transparent",
              border: canSend ? "1px solid var(--border-strong)" : "1px solid transparent",
              color: canSend ? "var(--text)" : "var(--muted)",
              opacity: canSend ? 1 : 0.4,
              cursor: canSend ? "pointer" : "not-allowed",
            }}
          >
            {isLoading ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Send size={16} />
            )}
          </button>
        </div>

        <p
          className="text-center text-[10px] mt-2"
          style={{ color: "var(--muted)" }}
        >
          Press Enter to run · Shift+Enter for newline
        </p>
      </div>
    </div>
  );
}
