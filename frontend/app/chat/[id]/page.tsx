"use client";

import { useEffect, useState, useCallback } from "react";
import { useRouter, useParams } from "next/navigation";
import { supabase } from "@/lib/supabase";
import { signOut } from "@/lib/auth";
import { getChats, getChat, createChat, sendQuery } from "@/lib/api";
import { Chat, Message, Dataset, ProcessingPhase, AssistantMetadata } from "@/types";
import Sidebar from "@/components/Sidebar";
import ChatPanel from "@/components/ChatPanel";
import UploadModal from "@/components/UploadModal";

const PHASES: ProcessingPhase[] = [
  "classifying",
  "retrieving",
  "generating",
  "executing",
  "evaluating",
  "enriching",
];

export default function ChatDetailPage() {
  const router = useRouter();
  const params = useParams();
  const chatId = params.id as string;

  const [chats, setChats]                     = useState<Chat[]>([]);
  const [messages, setMessages]               = useState<Message[]>([]);
  const [dataset, setDataset]                 = useState<Dataset | null>(null);
  const [userEmail, setUserEmail]             = useState("");
  const [token, setToken]                     = useState("");
  const [isLoading, setIsLoading]             = useState(false);
  const [processingPhase, setProcessingPhase] = useState<ProcessingPhase>("idle");
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [chatLoading, setChatLoading]         = useState(true);
  const [rateLimitMsg, setRateLimitMsg]       = useState("");

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) {
        router.replace("/login");
        return;
      }
      setToken(session.access_token);
      setUserEmail(session.user.email || "");
      loadChats(session.access_token);
      loadChat(chatId, session.access_token);
    });
  }, [chatId, router]);

  async function loadChats(tok: string) {
    try {
      const data = await getChats(tok);
      setChats(data);
    } catch {
      // ignore
    }
  }

  async function loadChat(id: string, tok: string) {
    setChatLoading(true);
    try {
      const data = await getChat(id, tok);
      setMessages(data.messages as Message[]);
      setDataset(data.dataset);
    } catch {
      // ignore
    } finally {
      setChatLoading(false);
    }
  }

  async function handleNewChat() {
    if (!token) return;
    try {
      const chat = await createChat(token);
      router.push(`/chat/${chat.id}`);
    } catch {
      // ignore
    }
  }

  async function handleSendQuery(query: string) {
    if (isLoading || !token) return;
    setIsLoading(true);

    // Optimistic user message
    const tempUserMsg: Message = {
      id: `temp-${Date.now()}`,
      chat_id: chatId,
      role: "user",
      content: query,
      metadata: null,
      sequence_number: messages.length + 1,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempUserMsg]);

    // Fake phase ticker
    let phaseIndex = 0;
    setProcessingPhase(PHASES[0]);
    const phaseInterval = setInterval(() => {
      phaseIndex++;
      if (phaseIndex < PHASES.length) {
        setProcessingPhase(PHASES[phaseIndex]);
      }
    }, 900);

    try {
      const result = await sendQuery(chatId, query, token);
      clearInterval(phaseInterval);
      setProcessingPhase("idle");

      const respData = result.response as any;

      const assistantMsg: Message = {
        id: result.message_id,
        chat_id: chatId,
        role: "assistant",
        content: respData.answer || "Analysis complete.",
        metadata: {
          answer:           respData.answer,
          insight:          respData.insight || "",
          anomalies:        respData.anomalies || [],
          follow_ups:       respData.follow_ups || [],
          confidence:       respData.confidence || "medium",
          code:             respData.code || "",
          result:           respData.result,
          result_type:      respData.result_type as AssistantMetadata["result_type"],
          mode:             respData.mode as AssistantMetadata["mode"],
          rag_context_used: respData.rag_context_used || [],
          retry_count:      respData.retry_count || 0,
        },
        sequence_number: messages.length + 2,
        created_at: result.created_at,
      };

      setMessages((prev) => [
        ...prev.filter((m) => m.id !== tempUserMsg.id),
        { ...tempUserMsg, id: `user-${Date.now()}` },
        assistantMsg,
      ]);

      loadChats(token);
    } catch (err: unknown) {
      clearInterval(phaseInterval);
      setProcessingPhase("idle");

      const axiosErr = err as { response?: { status?: number; data?: { detail?: string } } };
      let errorContent = "Connection failed. Please check your internet.";

      if (axiosErr?.response?.status === 504) {
        errorContent = "Analysis timed out. Please try again.";
      } else if (axiosErr?.response?.status === 429) {
        errorContent = "Rate limit reached. Please wait a moment before trying again.";
        setRateLimitMsg(errorContent);
        setTimeout(() => setRateLimitMsg(""), 5000);
      } else if (axiosErr?.response?.data?.detail) {
        errorContent = axiosErr.response.data.detail;
      }

      const errorMsg: Message = {
        id: `error-${Date.now()}`,
        chat_id: chatId,
        role: "assistant",
        content: errorContent,
        metadata: null,
        sequence_number: messages.length + 2,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  }

  function handleUploadSuccess(uploadedDataset: unknown) {
    setDataset(uploadedDataset as Dataset);
    setShowUploadModal(false);
  }

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "var(--bg)" }}>
      <Sidebar
        chats={chats}
        activeChatId={chatId}
        onSelectChat={(id) => router.push(`/chat/${id}`)}
        onNewChat={handleNewChat}
        onSignOut={signOut}
        userEmail={userEmail}
      />

      {/* Main content */}
      {chatLoading ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="space-y-3 w-full max-w-3xl px-5">
            <div className="skeleton rounded-xl" style={{ height: "80px" }} />
            <div className="skeleton rounded-xl" style={{ height: "160px" }} />
            <div className="skeleton rounded-xl" style={{ height: "120px" }} />
          </div>
        </div>
      ) : (
        <ChatPanel
          chatId={chatId}
          messages={messages}
          dataset={dataset}
          isLoading={isLoading}
          processingPhase={processingPhase}
          onSend={handleSendQuery}
          onUpload={() => setShowUploadModal(true)}
        />
      )}

      {/* Rate limit toast */}
      {rateLimitMsg && (
        <div
          className="fixed bottom-24 left-1/2 -translate-x-1/2 px-5 py-3 rounded-xl text-xs font-medium animate-fade-slide-in"
          style={{
            background: "var(--surface-2)",
            border: "1px solid var(--border-strong)",
            color: "var(--text)",
            boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
            opacity: 0,
            zIndex: 100,
          }}
        >
          {rateLimitMsg}
        </div>
      )}

      {/* Upload modal */}
      {showUploadModal && token && (
        <UploadModal
          chatId={chatId}
          token={token}
          onSuccess={handleUploadSuccess}
          onClose={() => setShowUploadModal(false)}
        />
      )}
    </div>
  );
}
