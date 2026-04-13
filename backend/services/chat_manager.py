import logging
import uuid
from datetime import datetime, timezone

from services.user_sync import ensure_user_exists

logger = logging.getLogger(__name__)


class ChatManager:
    def __init__(self, supabase):
        self.supabase = supabase

    def create_chat(self, user_id: str, email: str, title: str = "New Chat") -> dict:
        # Guarantee public.users row exists before the FK-dependent chat insert.
        # This eliminates the race condition between the auth trigger and this call.
        ensure_user_exists(self.supabase, user_id=user_id, email=email)

        chat_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        data = {
            "id": chat_id,
            "user_id": user_id,
            "title": title,
            "status": "active",
            "created_at": now,
            "updated_at": now,
        }
        result = self.supabase.table("chat").insert(data).execute()
        return result.data[0]

    def get_chats_by_user(self, user_id: str) -> list[dict]:
        result = (
            self.supabase.table("chat")
            .select("*")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
        )
        return result.data or []

    def get_chat_by_id(self, chat_id: str, user_id: str) -> dict:
        result = (
            self.supabase.table("chat")
            .select("*")
            .eq("id", chat_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            raise ValueError(f"Chat {chat_id} not found")
        return result.data[0]

    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        metadata: dict = None,
    ) -> dict:
        msg_result = (
            self.supabase.table("message")
            .select("sequence_number")
            .eq("chat_id", chat_id)
            .order("sequence_number", desc=True)
            .limit(1)
            .execute()
        )
        seq = 1
        if msg_result.data:
            seq = (msg_result.data[0].get("sequence_number") or 0) + 1

        message_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        msg_data = {
            "id": message_id,
            "chat_id": chat_id,
            "role": role,
            "content": content,
            # metadata is NOT NULL in the schema (DEFAULT '{}'::jsonb).
            # Guard against callers passing None to avoid a constraint violation.
            "metadata": metadata if metadata is not None else {},
            "sequence_number": seq,
            "created_at": now,
        }
        result = self.supabase.table("message").insert(msg_data).execute()
        self.supabase.table("chat").update({"updated_at": now}).eq(
            "id", chat_id
        ).execute()
        return result.data[0]

    def get_messages_by_chat(self, chat_id: str) -> list[dict]:
        result = (
            self.supabase.table("message")
            .select("*")
            .eq("chat_id", chat_id)
            .order("sequence_number")
            .execute()
        )
        return result.data or []

    def get_recent_context(self, chat_id: str, n: int = 10) -> list[dict]:
        result = (
            self.supabase.table("message")
            .select("role, content")
            .eq("chat_id", chat_id)
            .order("sequence_number", desc=True)
            .limit(n)
            .execute()
        )
        messages = result.data or []
        return list(reversed(messages))

    def auto_generate_title(self, first_query: str) -> str:
        if len(first_query) <= 50:
            return first_query
        truncated = first_query[:50]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            return truncated[:last_space]
        return truncated

    def update_title(self, chat_id: str, title: str) -> None:
        self.supabase.table("chat").update({"title": title}).eq(
            "id", chat_id
        ).execute()
