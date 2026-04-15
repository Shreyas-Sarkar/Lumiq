import uuid
from datetime import datetime, timezone


class ExecutionLogService:
    def __init__(self, supabase):
        self.supabase = supabase

    def log_execution(
        self,
        chat_id: str,
        message_id: str,
        user_id: str,
        query: str,
        code: str,
        output: str,
        status: str,
        error: str | None,
        retry_count: int,
        exec_time: float,
    ) -> None:
        try:
            log_id = str(uuid.uuid4())
            data = {
                "id": log_id,
                "chat_id": chat_id,
                "message_id": message_id,
                "user_id": user_id,
                "user_query": query[:2000],
                "generated_code": code[:5000],
                "execution_output": (output or "")[:2000],
                "execution_status": status,
                "error_message": error[:1000] if error else None,
                "retry_count": retry_count,
                "execution_time_ms": exec_time,
                "executed_at": datetime.now(timezone.utc).isoformat(),
            }
            self.supabase.table("execution_log").insert(data).execute()
        except Exception:
            pass

    def get_logs_by_chat(self, chat_id: str) -> list[dict]:
        try:
            result = (
                self.supabase.table("execution_log")
                .select("*")
                .eq("chat_id", chat_id)
                .order("executed_at", desc=True)
                .execute()
            )
            return result.data or []
        except Exception:
            return []
