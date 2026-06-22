from __future__ import annotations
import logging
import re
import time
from typing import Any, Callable, TypeVar

from app.config import Settings
from app.models.schemas import DocumentChunk, RetrievedContext
from app.models.types import ChunkId, DocId, EmbeddingVector, build_chunk_id

logger = logging.getLogger(__name__)

T = TypeVar("T")


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self._client = None
        self._index_name = settings.pinecone_index
        self._configured_index_name = settings.pinecone_index
        self._settings = settings
        self._index = None
        self._index_dimension: int | None = None

    @property
    def index_name(self) -> str:
        return self._index_name

    @property
    def index_dimension(self) -> int | None:
        return self._index_dimension

    def _retry_on_connection_error(
        self, func: Callable[..., T], max_retries: int = 3, base_delay: float = 1.0
    ) -> T:
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                # Check for port exhaustion or connection aborts
                err_msg = str(e).lower()
                is_conn_error = (
                    "can't assign requested address" in err_msg
                    or "connection aborted" in err_msg
                    or "protocolerror" in err_msg
                )

                if is_conn_error and attempt < max_retries - 1:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "Connection error during Pinecone operation (attempt %d/%d). Retrying in %.2fs... Error: %s",
                        attempt + 1,
                        max_retries,
                        delay,
                        e,
                    )
                    time.sleep(delay)
                    continue
                raise e
        # Should not be reachable
        raise RuntimeError("Retry loop exited unexpectedly")

    @staticmethod
    def _index_name_from_item(item: Any) -> str | None:
        if isinstance(item, dict):
            return item.get("name")
        return getattr(item, "name", None)

    @staticmethod
    def _index_dimension_from_item(item: Any) -> int | None:
        if isinstance(item, dict):
            dimension = item.get("dimension")
        else:
            dimension = getattr(item, "dimension", None)
        if dimension is None:
            return None
        return int(dimension)

    def _create_client(self):
        try:
            from pinecone import Pinecone
        except Exception as exc:
            raise RuntimeError(
                "Failed to import the official Pinecone SDK. "
                "Remove deprecated `pinecone-client` and install `pinecone[grpc]`."
            ) from exc
        return Pinecone(api_key=self._settings.pinecone_api_key)

    def _build_serverless_spec(self):
        try:
            from pinecone import ServerlessSpec
        except Exception:
            return None

        return ServerlessSpec(
            cloud=self._settings.pinecone_cloud,
            region=self._settings.pinecone_region,
        )

    def _ensure_index_exists(self, index_name: str, existing_names: set[str]) -> None:
        if index_name in existing_names:
            return

        logger.info("Creating Pinecone index: %s", index_name)
        self._client.create_index(
            name=index_name,
            dimension=self._settings.embedding_dim,
            metric=self._settings.pinecone_metric,
            spec=self._build_serverless_spec(),
        )
        existing_names.add(index_name)
        self._index_dimension = self._settings.embedding_dim

    # Complexity: Time O(1) | Space O(1)
    def ensure_index(self) -> None:
        if self._client is None:
            self._client = self._create_client()

        listed_indexes = list(self._client.list_indexes())
        index_items = {
            self._index_name_from_item(item): item
            for item in listed_indexes
            if self._index_name_from_item(item)
        }
        existing_names = set(index_items.keys())

        current_item = index_items.get(self._index_name)
        current_dimension = (
            self._index_dimension_from_item(current_item) if current_item else None
        )

        logger.info(
            "Available indexes: %s (target: %s, dim: %s)",
            list(index_items.keys()),
            self._index_name,
            self._settings.embedding_dim,
        )

        if current_item and current_dimension not in {
            None,
            self._settings.embedding_dim,
        }:
            raise ValueError(
                "Configured Pinecone index "
                f"'{self._index_name}' has dimension {current_dimension}, "
                f"but embedding model '{self._settings.embedding_model}' requires "
                f"{self._settings.embedding_dim}. Update PINECONE_INDEX or rebuild the index."
            )
        elif current_item is None:
            logger.warning(
                "Target index '%s' not found in Pinecone. Attempting to create it.",
                self._index_name,
            )
            self._ensure_index_exists(self._index_name, existing_names)
        else:
            logger.info("Using existing index '%s'", self._index_name)
            self._index_dimension = current_dimension or self._settings.embedding_dim

        self._index = self._client.Index(self._index_name)

    def _get_index(self):
        if self._index is None:
            self.ensure_index()
        return self._index

    @staticmethod
    def _extract_metadata(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return item.get("metadata", {})
        return getattr(item, "metadata", {}) or {}

    @staticmethod
    def _extract_vectors(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload.get("vectors", {})
        return getattr(payload, "vectors", {}) or {}

    # Complexity: Time O(n / batch_size) API calls | Space O(batch_size)
    def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        index = self._get_index()
        batch_size = self._settings.upsert_batch_size
        total_upserted = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vectors = [
                {
                    "id": str(chunk.chunk_id),
                    "values": chunk.embedding,
                    "metadata": {
                        "text": chunk.text,
                        "doc_id": str(chunk.doc_id),
                        "page": chunk.page,
                        "chunk_index": chunk.chunk_index,
                        "language": chunk.language.value,
                        "source_filename": chunk.source_filename,
                        **chunk.metadata,  # Include parent_text or other metadata
                    },
                }
                for chunk in batch
            ]

            self._retry_on_connection_error(lambda v=vectors: index.upsert(vectors=v))
            total_upserted += len(vectors)

        logger.info("Upserted %d vectors to Pinecone", total_upserted)
        return total_upserted

    # Complexity: Time O(1) API call | Space O(k)
    def query_similar(
        self,
        embedding: EmbeddingVector,
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
        sources: list[str] | None = None,
    ) -> list[RetrievedContext]:
        index = self._get_index()

        query_params: dict[str, Any] = {
            "vector": list(embedding),
            "top_k": top_k,
            "include_metadata": True,
        }

        # BM25 sparse scoring weight (applied in client-side reranker, not Pinecone)
        # Kept for forward-compat when index supports sparse vectors
        _ = self._settings.bm25_weight

        if filter_dict:
            query_params["filter"] = filter_dict
        elif sources:
            # Complexity: Time O(1) for filter construction
            query_params["filter"] = {"source_filename": {"$in": sources}}

        results = index.query(**query_params)

        contexts: list[RetrievedContext] = []
        for match in results.get("matches", []):
            meta = self._extract_metadata(match)
            context_text = meta.get("text", "")
            doc_id = meta.get("doc_id")
            chunk_index = meta.get("chunk_index")
            source_filename = meta.get("source_filename", "unknown")

            contexts.append(
                RetrievedContext(
                    text=context_text,
                    snippet=context_text,
                    score=float(match.get("score", 0.0)),
                    source_filename=source_filename,
                    page=int(meta.get("page", 0)),
                    chunk_id=ChunkId(match.get("id", "")),
                    doc_id=DocId(str(doc_id)) if doc_id else None,
                    chunk_index=int(chunk_index) if chunk_index is not None else None,
                )
            )

        logger.info(
            "Query returned %d contexts from %d sources. Top sources: %s",
            len(contexts),
            len(set(ctx.source_filename for ctx in contexts)),
            list(set(ctx.source_filename for ctx in contexts))[:5],
        )
        return contexts

    # Complexity: Time O(n) API fetch for neighboring chunks | Space O(n)
    def expand_contexts(
        self,
        contexts: list[RetrievedContext],
        radius: int,
    ) -> list[RetrievedContext]:
        if not contexts:
            return []

        # Group by (doc_id, page) to avoid cross-page contamination.
        # chunk_index is scoped per page; mixing pages produces wrong fetch IDs.
        groups: dict[tuple[str, int], set[int]] = {}
        doc_source: dict[tuple[str, int], str] = {}
        seed_contexts: dict[tuple[str, int], list[RetrievedContext]] = {}

        for ctx in contexts:
            if ctx.doc_id is None or ctx.chunk_index is None:
                continue
            key = (str(ctx.doc_id), ctx.page or 0)
            if key not in groups:
                groups[key] = set()
                doc_source[key] = ctx.source_filename or ""
                seed_contexts[key] = []
            seed_contexts[key].append(ctx)

            for idx in range(
                max(0, ctx.chunk_index - radius), ctx.chunk_index + radius + 1
            ):
                groups[key].add(idx)

        # Batch fetch all required chunks
        fetch_ids: list[str] = []
        for (doc_id, page), indices in groups.items():
            for idx in indices:
                fetch_ids.append(str(build_chunk_id(DocId(doc_id), page, idx)))

        if not fetch_ids:
            return contexts

        index = self._get_index()
        fetched = index.fetch(ids=fetch_ids)
        vectors = self._extract_vectors(fetched)

        # Build merged segments from the fetched vectors
        merged: list[RetrievedContext] = []
        for (doc_id, page), indices in groups.items():
            source_file = doc_source[(doc_id, page)]
            sorted_indices = sorted(indices)
            page_seed_contexts = seed_contexts.get((doc_id, page), [])

            if not sorted_indices:
                continue

            segment_parts: list[str] = []
            start_idx = sorted_indices[0]
            segment_indices: list[int] = []

            for i, idx in enumerate(sorted_indices):
                v_id = str(build_chunk_id(DocId(doc_id), page, idx))
                text = vectors.get(v_id, {}).get("metadata", {}).get("text", "")
                if text:
                    segment_parts.append(text.strip())
                    segment_indices.append(idx)

                # Flush on gap or end
                if i == len(sorted_indices) - 1 or sorted_indices[i + 1] != idx + 1:
                    full_text = " ".join(segment_parts)
                    cleaned_text = self._clean_internal_repetition(full_text)
                    supporting_contexts = [
                        ctx
                        for ctx in page_seed_contexts
                        if ctx.chunk_index is not None and ctx.chunk_index in segment_indices
                    ]
                    score = max((ctx.score for ctx in supporting_contexts), default=0.0)
                    rerank_score = max(
                        (
                            ctx.rerank_score
                            for ctx in supporting_contexts
                            if ctx.rerank_score is not None
                        ),
                        default=None,
                    )

                    if cleaned_text:
                        merged.append(
                            RetrievedContext(
                                chunk_id=f"{doc_id}_{page}_{start_idx}",
                                text=cleaned_text,
                                snippet=cleaned_text[:500],
                                score=score,
                                rerank_score=rerank_score,
                                source_filename=source_file,
                                page=page,
                                chunk_index=start_idx,
                                doc_id=DocId(doc_id),
                            )
                        )
                    segment_parts = []
                    segment_indices = []
                    if i < len(sorted_indices) - 1:
                        start_idx = sorted_indices[i + 1]

        return merged

    def _clean_internal_repetition(self, text: str) -> str:
        # Split by sentences and deduplicate contiguous duplicates
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            if not result or s_clean != result[-1]:
                # Only add if it's not a duplicate of the previous sentence
                result.append(s_clean)
        return " ".join(result)

    # Complexity: Time O(1) API call | Space O(1)
    def delete_by_doc_id(self, doc_id: DocId) -> None:
        index = self._get_index()
        index.delete(filter={"doc_id": {"$eq": str(doc_id)}})
        logger.info("Deleted vectors for doc_id: %s", doc_id)

    # Complexity: Time O(1) | Space O(1)
    def get_stats(self) -> dict[str, Any]:
        index = self._get_index()
        return index.describe_index_stats()

    def get_index_binding(self) -> dict[str, Any]:
        self._get_index()
        return {
            "configured_index": self._configured_index_name,
            "active_index": self._index_name,
            "embedding_dim": self._settings.embedding_dim,
            "index_dimension": self._index_dimension,
            "metric": self._settings.pinecone_metric,
        }

    # Complexity: Time O(1) | Space O(1)
    def doc_exists(self, doc_id: DocId) -> bool:
        index = self._get_index()

        def _check():
            results = index.query(
                vector=[0.0] * self._settings.embedding_dim,
                top_k=1,
                filter={"doc_id": {"$eq": str(doc_id)}},
                include_metadata=False,
            )
            return len(results.get("matches", [])) > 0

        return self._retry_on_connection_error(_check)
