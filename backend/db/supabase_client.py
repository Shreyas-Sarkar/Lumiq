from supabase import create_client, Client
from config import settings


def get_supabase_client() -> Client:
    if not settings.supabase_url or not settings.supabase_service_role_key:
        return None
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


try:
    supabase: Client = get_supabase_client()
except Exception:
    supabase = None
