"""
VectorStore — Qdrant adapter layer.

Provides a thin, backend-agnostic interface so the rest of the codebase
never touches Qdrant primitives directly.  RAGService is the only consumer.
"""

import os
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Cohere embed-english-v3.0 produces 1024-dimensional vectors
EMBEDDING_DIM = 1024
# Backward-compatible alias used by existing call sites.
VECTOR_SIZE = EMBEDDING_DIM


class VectorStore:
    """Low-level Qdrant adapter.  One instance is shared per process."""

    def __init__(self):
        self._client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        """Lazy-initialise the Qdrant client from environment variables."""
        if self._client is not None:
            return self._client

        try:
            from qdrant_client import QdrantClient

            url = os.getenv("QDRANT_URL", "")
            api_key = os.getenv("QDRANT_API_KEY", "")

            if not url:
                raise ValueError("QDRANT_URL is not set")

            self._client = QdrantClient(url=url, api_key=api_key or None)
            logger.info("Qdrant client initialised → %s", url)
        except Exception as exc:
            logger.error("Failed to create Qdrant client: %s", exc)
            raise RuntimeError(f"Qdrant unavailable: {exc}") from exc

        return self._client

    @staticmethod
    def _sanitise_name(name: str) -> str:
        """Qdrant collection names must not contain hyphens — replace with underscores."""
        return name.replace("-", "_")

    @staticmethod
    def _extract_collection_vector_size(collection_info: Any) -> int | None:
        """Best-effort extraction of vector size from Qdrant collection info."""
        try:
            vectors = collection_info.config.params.vectors
            # Single-vector config
            if hasattr(vectors, "size"):
                return int(vectors.size)
            # Named vectors config
            if isinstance(vectors, dict):
                first = next(iter(vectors.values()), None)
                if first is not None and hasattr(first, "size"):
                    return int(first.size)
        except Exception:
            return None
        return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def create_collection(self, name: str, vector_size: int = VECTOR_SIZE) -> None:
        """Create a Qdrant collection if it does not already exist."""
        from qdrant_client.models import Distance, VectorParams

        safe_name = self._sanitise_name(name)
        client = self._get_client()

        try:
            existing = {c.name for c in client.get_collections().collections}
            should_create = safe_name not in existing

            if not should_create:
                try:
                    info = client.get_collection(collection_name=safe_name)
                    existing_size = self._extract_collection_vector_size(info)
                    if existing_size is not None and existing_size != vector_size:
                        logger.warning(
                            "Collection '%s' has size=%s, expected=%s. Recreating.",
                            safe_name,
                            existing_size,
                            vector_size,
                        )
                        client.delete_collection(collection_name=safe_name)
                        should_create = True
                except Exception as exc:
                    logger.warning(
                        "Could not verify collection '%s' vector size: %s",
                        safe_name,
                        exc,
                    )

            if should_create:
                client.create_collection(
                    collection_name=safe_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("[QDRANT] Created collection '%s'", safe_name)
        except Exception as exc:
            logger.error("create_collection('%s') failed: %s", safe_name, exc)
            raise

    def upsert(
        self,
        collection: str,
        ids: list,
        embeddings: list,
        metadata: list,
    ) -> None:
        """
        Write points into a Qdrant collection.

        Parameters mirror the Chroma signature so callers need no changes:
            ids        — logical string IDs (e.g. 'schema', 'samples', 'stats')
            embeddings — list of float vectors
            metadata   — list of dicts used as Qdrant payload
        """
        from qdrant_client.models import PointStruct

        safe_name = self._sanitise_name(collection)
        client = self._get_client()

        if not embeddings or not metadata or not ids:
            logger.warning("upsert('%s') skipped: empty payload", safe_name)
            return

        if any((not emb or len(emb) != EMBEDDING_DIM) for emb in embeddings):
            logger.warning("upsert('%s') skipped: invalid embedding vector size", safe_name)
            return

        # Qdrant point IDs must be uint64 or UUID strings.
        # We store the original string ID inside the payload and use the
        # list-position integer as the Qdrant point ID.
        points = [
            PointStruct(
                id=idx,
                vector=embedding,
                payload={**meta, "_doc_id": doc_id},
            )
            for idx, (doc_id, embedding, meta) in enumerate(
                zip(ids, embeddings, metadata)
            )
        ]

        try:
            client.upsert(collection_name=safe_name, points=points)
            logger.info("[QDRANT] Upserted %d points → '%s'", len(points), safe_name)
        except Exception as exc:
            logger.error("upsert('%s') failed: %s", safe_name, exc)
            raise

    def query(
        self,
        collection: str,
        query_embedding: list,
        top_k: int = 3,
    ) -> list:
        """
        Search for the top-k nearest vectors.

        Returns a list of Qdrant ScoredPoint objects.
        Callers extract `.payload` from each hit.
        """
        safe_name = self._sanitise_name(collection)
        client = self._get_client()

        if not query_embedding or len(query_embedding) != EMBEDDING_DIM:
            logger.warning("query('%s') skipped: invalid query vector", safe_name)
            return []

        try:
            results = client.search(
                collection_name=safe_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
            )
            return results
        except Exception as exc:
            logger.error("query('%s') failed: %s", safe_name, exc)
            return []

    def delete_collection(self, name: str) -> None:
        """Delete a Qdrant collection; silently ignores missing collections."""
        safe_name = self._sanitise_name(name)
        client = self._get_client()

        try:
            client.delete_collection(collection_name=safe_name)
            logger.info("Deleted Qdrant collection '%s'", safe_name)
        except Exception as exc:
            logger.warning("delete_collection('%s') ignored: %s", safe_name, exc)

    def collection_exists(self, name: str) -> bool:
        """Return True if the collection exists in Qdrant."""
        safe_name = self._sanitise_name(name)
        try:
            client = self._get_client()
            existing = {c.name for c in client.get_collections().collections}
            return safe_name in existing
        except Exception:
            return False
