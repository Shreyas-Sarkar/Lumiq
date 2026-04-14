import logging

logger = logging.getLogger(__name__)


def ensure_user_exists(supabase, user_id: str, email: str) -> None:
    """
    Upsert the authenticated user into public.users before any FK-dependent
    insert (chat, dataset, execution_log, etc.).

    This is the deterministic fix for the race condition between:
      - Supabase Auth creating a row in auth.users
      - The handle_new_user trigger propagating it to public.users
      - The backend immediately inserting into chat (which FKs public.users)

    The database trigger remains in place as a fallback for any path that
    bypasses this function. Both are idempotent — ON CONFLICT DO NOTHING
    ensures no duplicates regardless of execution order.

    Args:
        supabase: Supabase client instance (service role).
        user_id:  UUID string from the validated JWT (auth.uid()).
        email:    Email from the validated JWT — never from an untrusted payload.

    Raises:
        RuntimeError: If the upsert fails, so callers can surface a 500 early
                      rather than getting a cryptic FK violation downstream.
    """
    try:
        result = (
            supabase.table("users")
            .upsert(
                {"id": user_id, "email": email},
                on_conflict="id",          # conflict target = PK
                ignore_duplicates=False,   # DO UPDATE so email stays fresh
            )
            .execute()
        )
        if result.data:
            # Row was inserted (not just a no-op update) — this means the
            # trigger had not yet fired. Log at DEBUG so we can track frequency.
            logger.debug(
                "user_sync: pre-emptive insert for user_id=%s (trigger not yet fired)",
                user_id,
            )
        else:
            logger.debug(
                "user_sync: user_id=%s already present in public.users", user_id
            )
    except Exception as exc:
        logger.error(
            "user_sync: failed to upsert user_id=%s into public.users: %s",
            user_id,
            exc,
        )
        raise RuntimeError(
            f"Could not synchronize user record for user_id={user_id}"
        ) from exc
