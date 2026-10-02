import copy
import uuid
from datetime import datetime, timezone
import pandas as pd
from services.analysis_tools import profile_dataset

OWNER = "11111111-1111-4111-8111-111111111111"
CHAT = "22222222-2222-4222-8222-222222222222"
DATASET = "33333333-3333-4333-8333-333333333333"
PLAN = {
    "title": "Revenue change",
    "steps": [
        {
            "operation": "compare_periods",
            "metric": "revenue",
            "date_column": "date",
            "group_by": "region",
            "periods": [["2026-01-01", "2026-01-31"], ["2026-02-01", "2026-02-28"]],
        }
    ],
}


class FakeStore:
    def __init__(self):
        self.rows = {}
        self.blocked = {}
        self.budgets = {}

    def get(self, rid, owner):
        r = self.rows.get(rid)
        return copy.deepcopy(r) if r and r["user_id"] == owner else None

    def list(self, chat, owner):
        return [
            copy.deepcopy(r)
            for r in self.rows.values()
            if r["chat_id"] == chat and r["user_id"] == owner
        ]

    def create(self, owner, chat, dataset, request_id, state, investigation_id=None):
        old = next(
            (
                r
                for r in self.rows.values()
                if r["user_id"] == owner and r["request_id"] == request_id
            ),
            None,
        )
        if old:
            if old["state"]["request_fingerprint"] != state["request_fingerprint"]:
                raise ValueError("Request ID already used")
            return copy.deepcopy(old)
        rid = str(uuid.uuid4())
        r = {
            "id": rid,
            "user_id": owner,
            "chat_id": chat,
            "dataset_id": dataset,
            "investigation_id": investigation_id or rid,
            "request_id": request_id,
            "state": copy.deepcopy(state),
            "revision": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.rows[rid] = r
        return copy.deepcopy(r)

    def cached_evidence(self, owner, dataset, key):
        return next(
            (
                copy.deepcopy(r["state"]["evidence"])
                for r in self.rows.values()
                if r["user_id"] == owner
                and r["dataset_id"] == dataset
                and r["state"].get("evidence_key") == key
                and r["state"].get("evidence")
            ),
            None,
        )

    def claim(self, rid, owner):
        r = self.get(rid, owner)
        if not r or self.rows[rid].get("lease_token"):
            return None
        r["lease_token"] = str(uuid.uuid4())
        self.rows[rid] = copy.deepcopy(r)
        return r

    def checkpoint(self, run, state, release=False):
        row = self.rows[run["id"]]
        if row.get("lease_token") != run.get("lease_token"):
            raise RuntimeError("Lost lease")
        row["state"] = copy.deepcopy(state)
        row["revision"] += 1
        if release:
            row["lease_token"] = None
        return copy.deepcopy(row)

    def reserve(self, provider, day, rpm, rpd, tpm, tokens):
        self.budgets[provider] = self.budgets.get(provider, 0) + 1
        return self.blocked.get(provider, {"allowed": True})

    def provider_failure(self, provider, retry_at=None, disabled=None):
        self.blocked[provider] = {
            "allowed": False,
            "reason": disabled or "quota",
            "retry_at": retry_at,
        }


class FakeDatasets:
    def __init__(self):
        self.loads = 0
        self.df = pd.read_csv("../samples/revenue.csv")

    def get_dataset_by_chat_id(self, chat, owner):
        if chat != CHAT or owner != OWNER:
            raise ValueError("Not found")
        return {
            "id": DATASET,
            "storage_path": "fake://durable",
            "content_hash": "immutable",
            "profile": profile_dataset(self.df),
        }

    def load_dataframe_from_storage(self, path, owner=None):
        self.loads += 1
        return self.df.copy()


class FakeChats:
    def get_chat_by_id(self, chat, owner):
        if owner != OWNER or chat != CHAT:
            raise ValueError("Not found")
        return {"id": chat}
