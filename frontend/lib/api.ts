import axios from "axios";
import { Chat, Message, Dataset } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
});

export async function createChat(token: string): Promise<Chat> {
  const res = await api.post(
    "/api/chats",
    { title: "New Chat" },
    { headers: { Authorization: `Bearer ${token}` } }
  );
  return res.data;
}

export async function getChats(token: string): Promise<Chat[]> {
  const res = await api.get("/api/chats", {
    headers: { Authorization: `Bearer ${token}` },
  });
  return res.data;
}

export async function getChat(
  chatId: string,
  token: string
): Promise<{ chat: Chat; messages: Message[]; dataset: Dataset | null }> {
  const res = await api.get(`/api/chat/${chatId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return res.data;
}

export async function sendQuery(
  chatId: string,
  query: string,
  token: string
): Promise<{
  message_id: string;
  response: {
    answer: string;
    code: string;
    result: unknown;
    result_type: string;
    reasoning: string;
    assumptions: string[];
    rag_context_used: string[];
    retry_count: number;
  };
  chat_id: string;
  created_at: string;
}> {
  const res = await api.post(
    "/api/chat",
    { chat_id: chatId, query },
    { headers: { Authorization: `Bearer ${token}` } }
  );
  return res.data;
}

export async function uploadDataset(
  chatId: string,
  file: File,
  token: string
): Promise<Dataset> {
  const formData = new FormData();
  formData.append("chat_id", chatId);
  formData.append("file", file);
  const res = await api.post("/api/upload", formData, {
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "multipart/form-data",
    },
  });
  return res.data.dataset;
}
