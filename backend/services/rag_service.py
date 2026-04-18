"""
RAGService — Retrieval-Augmented Generation helper.

Public interface is IDENTICAL to the previous ChromaDB implementation so
no call-site changes are required in the orchestrator, upload router, or
health check:

    _ensure_initialized()
    get_or_create_collection(collection_id)   [internal — kept for compat]
    build_documents(schema, samples, stats)
    index_dataset(collection_id, schema, samples, stats)
    retrieve_context(collection_id, query, n_results)
    delete_collection(collection_id)

Storage backend swapped: ChromaDB → Qdrant (via VectorStore adapter).
Embedding backend migrated to Cohere API (dim=1024).
"""

import json
import logging
from typing import Optional

from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# Logical document IDs — stored in Qdrant payload so we can identify each chunk.
_DOC_IDS = ["schema", "samples", "stats"]


class RAGService:
    # ------------------------------------------------------------------
    # Qdrant-backed implementation
    # The constructor signature is kept parameter-compatible with the old
    # ChromaDB version (persist_dir= is accepted but unused — Qdrant is
    # remote and needs no local filesystem directory).
    # ------------------------------------------------------------------

    def __init__(self, persist_dir: Optional[str] = None):
        # persist_dir is retained so main.py and any future call sites that
        # pass it as a keyword argument continue to work without modification.
        self._persist_dir = persist_dir  # not used
        self._embedding_service = None
        self._vector_store = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ensure_initialized(self):
        """Lazy-load the embedding client and Qdrant client."""
        if self._initialized:
            return

        try:
            from services.vector_store import VectorStore

            self._embedding_service = EmbeddingService()
            self._vector_store = VectorStore()
            # Force a connection probe so errors surface early.
            self._vector_store._get_client()

            self._initialized = True
            logger.info("RAGService initialised (Qdrant backend)")
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise RAGService: {exc}") from exc

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Return a list of float vectors for the given texts."""
        try:
            return self._embedding_service.embed(texts)
        except Exception as exc:
            logger.warning("Embedding generation failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Collection management (kept for backward-compat)
    # ------------------------------------------------------------------

    def get_or_create_collection(self, collection_id: str):
        """
        Ensure the Qdrant collection exists and return its sanitised name.

        Previously returned a Chroma Collection object; now returns the
        sanitised string name.  Internal callers only used the name to route
        further Chroma calls, so nothing breaks.
        """
        self._ensure_initialized()
        from services.vector_store import VECTOR_SIZE

        self._vector_store.create_collection(collection_id, vector_size=VECTOR_SIZE)
        return self._vector_store._sanitise_name(collection_id)

    # ------------------------------------------------------------------
    # Document construction (unchanged logic)
    # ------------------------------------------------------------------

    def build_documents(
        self, schema: dict, samples: list, stats: dict
    ) -> list[str]:
        schema_lines = ["Dataset schema:"]
        for col in schema.get("columns", []):
            line = f"  - {col['name']} ({col['dtype']})"
            if col.get("nullable"):
                line += " [nullable]"
            if "unique_values" in col:
                line += f" values: {col['unique_values']}"
            schema_lines.append(line)
        schema_doc = "\n".join(schema_lines)

        sample_lines = ["Sample data rows:"]
        for i, row in enumerate(samples[:5]):
            sample_lines.append(f"  Row {i + 1}: {json.dumps(row, default=str)}")
        sample_doc = "\n".join(sample_lines)

        stats_lines = ["Statistical summary:"]
        meta = stats.get("_meta", {})
        if meta:
            stats_lines.append(
                f"  Total rows: {meta.get('total_rows')}, "
                f"Total columns: {meta.get('total_columns')}"
            )
        for col, col_stats in stats.items():
            if col == "_meta" or not isinstance(col_stats, dict):
                continue
            if "mean" in col_stats or "count" in col_stats:
                parts = []
                for k in ["count", "mean", "min", "max", "std"]:
                    if col_stats.get(k) is not None:
                        parts.append(
                            f"{k}={col_stats[k]:.2f}"
                            if isinstance(col_stats[k], float)
                            else f"{k}={col_stats[k]}"
                        )
                stats_lines.append(f"  {col}: {', '.join(parts)}")
        stats_doc = "\n".join(stats_lines)

        return [schema_doc, sample_doc, stats_doc]

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_dataset(
        self, collection_id: str, schema: dict, samples: list, stats: dict
    ):
        """
        Embed and upsert the three dataset context documents into Qdrant.

        Replaces any existing points for this collection (delete + re-upsert)
        so re-uploads behave identically to the previous Chroma implementation.
        """
        self._ensure_initialized()

        documents = self.build_documents(schema, samples, stats)
        embeddings = self._embed(documents)

        # Delete stale collection first (idempotent on first run)
        try:
            self._vector_store.delete_collection(collection_id)
        except Exception:
            pass

        # Re-create and upsert fresh
        from services.vector_store import VECTOR_SIZE

        self._vector_store.create_collection(collection_id, vector_size=VECTOR_SIZE)

        metadata = [{"text": doc, "doc_id": doc_id} for doc, doc_id in zip(documents, _DOC_IDS)]
        self._vector_store.upsert(
            collection=collection_id,
            ids=_DOC_IDS,
            embeddings=embeddings,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Retrieval  — returns list[str] exactly as Chroma did
    # ------------------------------------------------------------------

    def retrieve_context(
        self, collection_id: str, query: str, n_results: int = 3
    ) -> list[str]:
        """
        Semantic search over the dataset context chunks.

        Returns list[str] — the same shape as:
            results["documents"][0]   (Chroma's response format)
        Callers are unchanged.
        """
        self._ensure_initialized()
        try:
            query_vector = self._embed([query])[0]
            hits = self._vector_store.query(
                collection=collection_id,
                query_embedding=query_vector,
                top_k=n_results,
            )
            # Extract the stored text from each hit's payload
            return [hit.payload.get("text", "") for hit in hits if hit.payload]
        except Exception as exc:
            logger.warning("retrieve_context failed for '%s': %s", collection_id, exc)
            return []

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_collection(self, collection_id: str):
        """Remove a dataset's vector collection from Qdrant."""
        self._ensure_initialized()
        try:
            self._vector_store.delete_collection(collection_id)
        except Exception:
            pass
