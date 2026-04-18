"use client";

import { formatDistanceToNow } from "date-fns";
import { Plus, Database, LogOut, BarChart3 } from "lucide-react";
import { Chat } from "@/types";

interface SidebarProps {
  chats: Chat[];
  activeChatId: string | null;
  onSelectChat: (chatId: string) => void;
  onNewChat: () => void;
  onSignOut: () => void;
  userEmail: string;
  isLoading?: boolean;
}

export default function Sidebar({
  chats,
  activeChatId,
  onSelectChat,
  onNewChat,
  onSignOut,
  userEmail,
  isLoading,
}: SidebarProps) {
  return (
    <div
      className="w-56 flex-shrink-0 flex flex-col h-full"
      style={{
        background: "var(--surface)",
        borderRight: "1px solid var(--border)",
      }}
    >
      {/* ── Brand ── */}
      <div
        className="px-4 pt-5 pb-4"
        style={{ borderBottom: "1px solid var(--border)" }}
      >
        <div className="flex items-center gap-2 mb-4">
          <div
            className="w-6 h-6 rounded flex items-center justify-center text-[11px] font-bold flex-shrink-0"
            style={{
              background: "var(--surface-3)",
              border: "1px solid var(--border-strong)",
              color: "var(--text)",
            }}
          >
            L
          </div>
          <span className="font-semibold text-sm" style={{ color: "var(--text)" }}>
            Lumiq
          </span>
        </div>
        <button
          onClick={onNewChat}
          id="new-analysis-btn"
          className="w-full flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-medium transition-all duration-150"
          style={{
            background: "var(--surface-2)",
            border: "1px solid var(--border-strong)",
            color: "var(--text)",
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLButtonElement).style.background = "var(--surface-3)";
            (e.currentTarget as HTMLButtonElement).style.borderColor = "rgba(255,255,255,0.15)";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLButtonElement).style.background = "var(--surface-2)";
            (e.currentTarget as HTMLButtonElement).style.borderColor = "var(--border-strong)";
          }}
        >
          <Plus size={13} />
          New analysis
        </button>
      </div>

      {/* ── Section label ── */}
      <div className="px-4 pt-4 pb-2">
        <span
          className="text-[10px] uppercase tracking-widest font-medium"
          style={{ color: "var(--muted)" }}
        >
          Analyses
        </span>
      </div>

      {/* ── Chat list ── */}
      <div className="flex-1 overflow-y-auto px-2 pb-2">
        {isLoading ? (
          <div className="space-y-1.5 px-2 pt-1">
            {[1, 2, 3].map((i) => (
              <div key={i} className="skeleton rounded-lg h-10" />
            ))}
          </div>
        ) : chats.length === 0 ? (
          <div className="px-4 py-6 text-center">
            <BarChart3 size={22} className="mx-auto mb-2 opacity-20" style={{ color: "var(--muted)" }} />
            <p className="text-xs" style={{ color: "var(--muted)" }}>
              No analyses yet
            </p>
          </div>
        ) : (
          chats.map((chat) => {
            const isActive = chat.id === activeChatId;
            return (
              <button
                key={chat.id}
                onClick={() => onSelectChat(chat.id)}
                className="w-full text-left rounded-lg px-3 py-2.5 mb-0.5 transition-all duration-100 group"
                style={{
                  background: isActive ? "var(--surface-2)" : "transparent",
                  border: isActive
                    ? "1px solid var(--border-strong)"
                    : "1px solid transparent",
                }}
                onMouseEnter={(e) => {
                  if (!isActive) {
                    (e.currentTarget as HTMLButtonElement).style.background =
                      "var(--surface-2)";
                    (e.currentTarget as HTMLButtonElement).style.borderColor =
                      "var(--border)";
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isActive) {
                    (e.currentTarget as HTMLButtonElement).style.background =
                      "transparent";
                    (e.currentTarget as HTMLButtonElement).style.borderColor =
                      "transparent";
                  }
                }}
              >
                <div className="flex items-center gap-2">
                  {chat.dataset_id ? (
                    <Database
                      size={11}
                      className="flex-shrink-0"
                      style={{ color: isActive ? "var(--accent)" : "var(--muted)" }}
                    />
                  ) : (
                    <BarChart3
                      size={11}
                      className="flex-shrink-0"
                      style={{ color: "var(--muted)" }}
                    />
                  )}
                  <span
                    className="text-xs truncate"
                    style={{
                      color: isActive ? "var(--text)" : "var(--accent)",
                      fontWeight: isActive ? 500 : 400,
                    }}
                  >
                    {chat.title}
                  </span>
                </div>
                <div
                  className="text-[10px] mt-0.5 pl-4"
                  style={{ color: "var(--muted)" }}
                >
                  {formatDistanceToNow(new Date(chat.updated_at), {
                    addSuffix: true,
                  })}
                </div>
              </button>
            );
          })
        )}
      </div>

      {/* ── User ── */}
      <div
        className="px-4 py-3 flex items-center gap-2"
        style={{ borderTop: "1px solid var(--border)" }}
      >
        <div
          className="w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-semibold flex-shrink-0"
          style={{
            background: "var(--surface-3)",
            border: "1px solid var(--border-strong)",
            color: "var(--text)",
          }}
        >
          {userEmail.charAt(0).toUpperCase() || "U"}
        </div>
        <span className="text-xs truncate flex-1" style={{ color: "var(--muted)" }}>
          {userEmail}
        </span>
        <button
          onClick={onSignOut}
          title="Sign out"
          className="flex-shrink-0 transition-colors"
          style={{ color: "var(--muted)" }}
          onMouseEnter={(e) =>
            ((e.currentTarget as HTMLButtonElement).style.color = "var(--text)")
          }
          onMouseLeave={(e) =>
            ((e.currentTarget as HTMLButtonElement).style.color = "var(--muted)")
          }
        >
          <LogOut size={13} />
        </button>
      </div>
    </div>
  );
}
