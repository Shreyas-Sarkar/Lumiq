"""Owner-scoped PostgREST checkpoints and atomic leases; no direct DB connection needed."""

import uuid


class RunStore:
    def __init__(self, supabase):
        self.db = supabase

    def recoverable(self):
        return (
            self.db.table("investigation_run")
            .select("*")
            .in_(
                "state->>status",
                ["queued", "planning", "computing", "interpreting", "interrupted"],
            )
            .order("created_at")
            .limit(3)
            .execute()
            .data
        )

    def get(self, run_id, owner):
        rows = (
            self.db.table("investigation_run")
            .select("*")
            .eq("id", run_id)
            .eq("user_id", owner)
            .limit(1)
            .execute()
            .data
        )
        return rows[0] if rows else None

    def list(self, chat_id, owner):
        return (
            self.db.table("investigation_run")
            .select("*")
            .eq("chat_id", chat_id)
            .eq("user_id", owner)
            .order("created_at", desc=True)
            .limit(50)
            .execute()
            .data
        )

    def create(
        self, owner, chat_id, dataset_id, request_id, state, investigation_id=None
    ):
        existing = (
            self.db.table("investigation_run")
            .select("*")
            .eq("user_id", owner)
            .eq("request_id", request_id)
            .limit(1)
            .execute()
            .data
        )
        if existing:
            if (
                existing[0]["chat_id"] != chat_id
                or existing[0]["state"].get("request_fingerprint")
                != state["request_fingerprint"]
            ):
                raise ValueError("Request ID already used for another question")
            return existing[0]
        data = {
            "id": str(uuid.uuid4()),
            "user_id": owner,
            "chat_id": chat_id,
            "dataset_id": dataset_id,
            "request_id": request_id,
            "investigation_id": investigation_id or str(uuid.uuid4()),
            "state": state,
        }
        try:
            return self.db.table("investigation_run").insert(data).execute().data[0]
        except Exception:
            # Resolve only actual idempotency races; never hide database failures.
            existing = (
                self.db.table("investigation_run")
                .select("*")
                .eq("user_id", owner)
                .eq("request_id", request_id)
                .limit(1)
                .execute()
                .data
            )
            if (
                existing
                and existing[0]["chat_id"] == chat_id
                and existing[0]["state"].get("request_fingerprint")
                == state["request_fingerprint"]
            ):
                return existing[0]
            raise

    def cached_evidence(self, owner, dataset_id, key):
        rows = (
            self.db.table("investigation_run")
            .select("state")
            .eq("user_id", owner)
            .eq("dataset_id", dataset_id)
            .contains("state", {"evidence_key": key})
            .limit(1)
            .execute()
            .data
        )
        return rows[0]["state"].get("evidence") if rows else None

    def cached_interpretation(self, owner, dataset_id, key):
        rows = (
            self.db.table("investigation_run")
            .select("state")
            .eq("user_id", owner)
            .eq("dataset_id", dataset_id)
            .contains("state", {"status": "complete", "interpretation_key": key})
            .limit(1)
            .execute()
            .data
        )
        return rows[0]["state"].get("interpretation_payload") if rows else None

    def claim(self, run_id, owner):
        token = str(uuid.uuid4())
        rows = (
            self.db.rpc(
                "claim_investigation",
                {"p_id": run_id, "p_owner": owner, "p_token": token},
            )
            .execute()
            .data
        )
        return rows[0] if rows else None

    def checkpoint(self, run, state, release=False):
        rows = (
            self.db.rpc(
                "checkpoint_investigation",
                {
                    "p_id": run["id"],
                    "p_owner": run["user_id"],
                    "p_token": run["lease_token"],
                    "p_state": state,
                    "p_release": release,
                },
            )
            .execute()
            .data
        )
        if not rows:
            raise RuntimeError("Run lease lost")
        return rows[0]

    def reserve(self, provider, day, rpm, rpd, tpm, tokens):
        return (
            self.db.rpc(
                "reserve_provider",
                {
                    "p_provider": provider,
                    "p_day": day,
                    "p_rpm": rpm,
                    "p_rpd": rpd,
                    "p_tpm": tpm,
                    "p_tokens": tokens,
                },
            )
            .execute()
            .data
        )

    def provider_failure(self, provider, retry_at=None, disabled=None):
        data = {"provider": provider}
        if retry_at:
            data["cooldown_until"] = retry_at
        if disabled:
            data["disabled_reason"] = disabled
        self.db.table("provider_budget").upsert(data, on_conflict="provider").execute()
