"use client";
import { useParams } from "next/navigation";
import InvestigationWorkspace from "@/components/InvestigationWorkspace";
export default function WorkspacePage() {
  const params = useParams<{ id: string }>();
  return <InvestigationWorkspace chatId={params.id} />;
}
