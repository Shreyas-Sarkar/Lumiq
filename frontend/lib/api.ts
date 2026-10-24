import axios from "axios";
import { Chat, Message, Dataset } from "@/types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export async function createChat(token: string): Promise<Chat> {
  const res = await api.post(
    "/api/chats",
    { title: "New Chat" },
    { headers: { Authorization: `Bearer ${token}` } },
  );
  return res.data;
}

export async function renameWorkspace(
  chatId: string,
  title: string,
  token: string,
): Promise<Chat> {
  const res = await api.post(
    `/api/chats/${chatId}/rename`,
    { title },
    {
      headers: { Authorization: `Bearer ${token}` },
    },
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
  token: string,
): Promise<{ chat: Chat; messages: Message[]; dataset: Dataset | null }> {
  const res = await api.get(`/api/chat/${chatId}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return res.data;
}

export async function sendQuery(
  chatId: string,
  query: string,
  token: string,
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
    { headers: { Authorization: `Bearer ${token}` } },
  );
  return res.data;
}

export async function uploadDataset(
  chatId: string,
  file: File,
  token: string,
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

export async function getProfile(chatId: string, token: string) {
  return (
    await api.get<import("@/types/investigations").DatasetProfile>(
      `/api/datasets/${chatId}/profile`,
      { headers: { Authorization: `Bearer ${token}` } },
    )
  ).data;
}
export async function getInvestigations(chatId: string, token: string) {
  return (
    await api.get<import("@/types/investigations").Investigation[]>(
      "/api/investigations",
      {
        params: { chat_id: chatId },
        headers: { Authorization: `Bearer ${token}` },
      },
    )
  ).data;
}
export async function getInvestigation(id: string, token: string) {
  return (
    await api.get<import("@/types/investigations").Investigation>(
      `/api/investigations/${id}`,
      { headers: { Authorization: `Bearer ${token}` } },
    )
  ).data;
}
export async function startInvestigation(
  body: {
    chat_id: string;
    question: string;
    request_id: string;
    parent_id?: string;
    plan?: import("@/types/investigations").AnalysisPlan;
  },
  token: string,
) {
  return (
    await api.post<import("@/types/investigations").Investigation>(
      "/api/investigations",
      body,
      { headers: { Authorization: `Bearer ${token}` } },
    )
  ).data;
}
export async function resumeInvestigation(
  id: string,
  body: { clarification?: string },
  token: string,
) {
  return (
    await api.post<import("@/types/investigations").Investigation>(
      `/api/investigations/${id}/resume`,
      body,
      { headers: { Authorization: `Bearer ${token}` } },
    )
  ).data;
}
export async function downloadEvidence(
  id: string,
  evidenceId: string,
  token: string,
) {
  const response = await api.get(
    `/api/investigations/${id}/evidence/${evidenceId}/download`,
    { responseType: "blob", headers: { Authorization: `Bearer ${token}` } },
  );
  const url = URL.createObjectURL(response.data);
  const a = document.createElement("a");
  a.href = url;
  a.download = `evidence-${evidenceId}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}
export function apiError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    if (error.response?.status === 401 || error.response?.status === 403)
      return "Your session expired. Sign in again to continue.";
    const detail = error.response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (error.code === "ECONNABORTED")
      return "The backend is taking longer to respond. It may be waking up. Retry to reload saved progress.";
    if (
      !error.response &&
      /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(?::\d+)?(?:\/|$)/i.test(
        API_BASE,
      )
    )
      return `The local API at ${API_BASE} is unreachable. Start the backend, then retry to reload saved progress.`;
  }
  return "Could not reach the backend. It may be waking up or offline. Retry to reload saved progress.";
}
