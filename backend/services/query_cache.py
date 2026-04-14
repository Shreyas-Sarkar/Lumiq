import hashlib
import time
import os
from typing import Optional


class QueryCache:
    """
    In-memory LRU-style cache keyed by (query, dataset_hash).

    Caches both the execution result serialization and the cognitive engine
    output so that identical queries on the same dataset skip all computation
    and LLM calls.

    TTL defaults to 1 hour. Max size defaults to 200 entries (oldest evicted).
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 200):
        self._cache: dict[str, dict] = {}
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str, dataset_hash: str) -> Optional[dict]:
        key = self._make_key(query, dataset_hash)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if (time.time() - entry["ts"]) >= self.ttl_seconds:
            del self._cache[key]
            return None
        return entry["data"]

    def set(self, query: str, dataset_hash: str, data: dict) -> None:
        if len(self._cache) >= self.max_size:
            # Evict the oldest entry
            oldest_key = min(self._cache, key=lambda k: self._cache[k]["ts"])
            del self._cache[oldest_key]
        key = self._make_key(query, dataset_hash)
        self._cache[key] = {"data": data, "ts": time.time()}

    def invalidate(self, dataset_hash: str) -> None:
        """Remove all entries for a given dataset (e.g., after re-upload)."""
        keys_to_del = [k for k, v in self._cache.items() if dataset_hash in k]
        for k in keys_to_del:
            del self._cache[k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_key(self, query: str, dataset_hash: str) -> str:
        raw = f"{query.strip().lower()}::{dataset_hash}"
        return hashlib.md5(raw.encode()).hexdigest()

    @staticmethod
    def compute_dataset_hash(storage_path: str) -> str:
        """
        Derive a stable short hash from the file's path + size + mtime.
        This is fast (no file read) and changes whenever the file changes.
        """
        try:
            stat = os.stat(storage_path)
            raw = f"{storage_path}::{stat.st_size}::{stat.st_mtime}"
        except OSError:
            raw = storage_path
        return hashlib.md5(raw.encode()).hexdigest()[:16]
