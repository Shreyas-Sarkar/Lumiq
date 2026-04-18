"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";
import { signOut } from "@/lib/auth";
import { getChats, createChat } from "@/lib/api";
import { Chat } from "@/types";
import Sidebar from "@/components/Sidebar";
import { BarChart3, ArrowRight } from "lucide-react";

export default function ChatPage() {
  const router = useRouter();
  const [chats, setChats] = useState<Chat[]>([]);
  const [userEmail, setUserEmail] = useState("");
  const [token, setToken] = useState("");
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) {
        router.replace("/login");
        return;
      }
      setToken(session.access_token);
      setUserEmail(session.user.email || "");
      loadChats(session.access_token);
    });
  }, [router]);

  async function loadChats(tok: string) {
    try {
      const data = await getChats(tok);
      setChats(data);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }

  async function handleNewChat() {
    if (!token || creating) return;
    setCreating(true);
    try {
      const chat = await createChat(token);
      router.push(`/chat/${chat.id}`);
    } catch {
      // ignore
    } finally {
      setCreating(false);
    }
  }

  function handleSelectChat(chatId: string) {
    router.push(`/chat/${chatId}`);
  }

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "var(--bg)" }}>
      <Sidebar
        chats={chats}
        activeChatId={null}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        onSignOut={signOut}
        userEmail={userEmail}
        isLoading={loading}
      />

      {/* Empty state */}
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center max-w-sm px-6">
          <div
            className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-6"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border-strong)",
            }}
          >
            <BarChart3 size={26} style={{ color: "var(--muted)" }} />
          </div>
          <h2 className="text-base font-semibold mb-2" style={{ color: "var(--text)" }}>
            Start a new analysis
          </h2>
          <p className="text-sm leading-relaxed mb-6" style={{ color: "var(--muted)" }}>
            Upload a CSV and ask questions. Every answer is grounded in real code execution — no guesswork.
          </p>
          <button
            onClick={handleNewChat}
            disabled={creating}
            id="new-analysis-home-btn"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all duration-150"
            style={{
              background: creating ? "var(--surface-2)" : "var(--text)",
              color: creating ? "var(--muted)" : "var(--bg)",
              border: "1px solid transparent",
              cursor: creating ? "wait" : "pointer",
            }}
          >
            {creating ? "Creating…" : (
              <>
                New analysis <ArrowRight size={14} />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
