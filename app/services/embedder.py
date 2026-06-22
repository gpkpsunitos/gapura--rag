from __future__ import annotations

import hashlib
import logging
import os
import re
from functools import lru_cache
from typing import Sequence

import numpy as np
from huggingface_hub import InferenceClient

from app.models.types import EmbeddingVector

logger = logging.getLogger(__name__)

_FALLBACK_DIM = 768
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_client: InferenceClient | None = None
_embedding_mode_cache: dict[str, str] = {}


def _get_client() -> InferenceClient:
    global _client
    if _client is None:
        _client = InferenceClient(token=os.getenv("HF_TOKEN"))
    return _client


def _hash_encode(
    texts: list[str],
    dim: int = _FALLBACK_DIM,
) -> list[EmbeddingVector]:
    vectors: list[EmbeddingVector] = []

    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            primary = int.from_bytes(digest[:4], "big") % dim
            secondary = int.from_bytes(digest[4:8], "big") % dim
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vec[primary] += sign
            vec[secondary] += sign * 0.5

        if not np.any(vec):
            vec[0] = 1.0

        norm = np.linalg.norm(vec)
        if norm:
            vec /= norm
        vectors.append(EmbeddingVector(vec.tolist()))

    return vectors


def _feature_extract_encode(
    texts: list[str],
    model_name: str,
) -> list[EmbeddingVector]:
    client = _get_client()
    if len(texts) == 1:
        raw = client.feature_extraction(
            text=texts[0],
            model=model_name,
            normalize=True,
        )
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        if arr.ndim != 1:
            raise ValueError(f"Unexpected single embedding shape {arr.shape}")
        norm = np.linalg.norm(arr)
        if norm:
            arr /= norm
        return [EmbeddingVector(arr.tolist())]

    # Batch call to Inference API is much faster than N sequential calls
    raw = client.feature_extraction(
        text=texts,
        model=model_name,
        normalize=True,
    )

    # Initial raw response could be (batch, dim) or (batch, seq, dim)
    full_array = np.asarray(raw, dtype=np.float32)

    # Ensure we get (batch, dim)
    if full_array.ndim == 3:
        # Mean pooling over tokens (axis 1)
        full_array = full_array.mean(axis=1)

    if full_array.ndim != 2 or full_array.shape[0] != len(texts):
        # Fallback to loop if shape is unexpected when batched
        logger.warning(
            "Unexpected batch embedding shape %s, falling back to sequential",
            full_array.shape,
        )
        vectors: list[EmbeddingVector] = []
        for text in texts:
            items = client.feature_extraction(text=text, model=model_name, normalize=True)
            arr = np.asarray(items, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            norm = np.linalg.norm(arr)
            if norm:
                arr /= norm
            vectors.append(EmbeddingVector(arr.tolist()))
        return vectors

    # Normalize each vector in the batch
    norms = np.linalg.norm(full_array, axis=1, keepdims=True)
    # Avoid division by zero
    full_array = np.divide(full_array, norms, out=full_array, where=norms > 0)

    return [EmbeddingVector(row.tolist()) for row in full_array]


@lru_cache(maxsize=512)
def _feature_extract_encode_single_cached(
    text: str,
    model_name: str,
) -> tuple[float, ...]:
    vector = _feature_extract_encode([text], model_name)[0]
    return tuple(vector)


def preload_models(model_name: str) -> None:
    # We use hosted inference for embeddings, so there is nothing to preload locally.
    _embedding_mode_cache.setdefault(model_name, "remote")


def embed_passages(
    texts: Sequence[str],
    model_name: str,
    batch_size: int = 32,
) -> list[EmbeddingVector]:
    if not texts:
        return []

    # Apply "passage: " prefix as required by some multilingual models (e.g. E5 family)
    prefixed = [f"passage: {t}" for t in texts]
    
    # Send in batches to avoid API payload limits (HTTP 413) and timeouts
    all_vectors: list[EmbeddingVector] = []
    for i in range(0, len(prefixed), batch_size):
        batch = prefixed[i : i + batch_size]
        logger.info("Embedding passage batch %d/%d (size %d)", i // batch_size + 1, (len(prefixed) - 1) // batch_size + 1, len(batch))
        all_vectors.extend(_encode(batch, model_name))
        
    return all_vectors


def embed_query(
    text: str,
    model_name: str,
) -> EmbeddingVector:
    prefixed = f"query: {text}"
    mode = _embedding_mode_cache.get(model_name, "remote")
    if mode == "hash":
        return _hash_encode([prefixed])[0]

    try:
        vector = _feature_extract_encode_single_cached(prefixed, model_name)
        _embedding_mode_cache[model_name] = "remote"
        return EmbeddingVector(list(vector))
    except Exception as exc:
        if "doesn't support task 'feature-extraction'" in str(exc):
            logger.warning(
                "Embedding model '%s' is not compatible with feature_extraction on Hugging Face Inference. "
                "Set EMBEDDING_MODEL to a feature-extraction model such as "
                "'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'. "
                "Falling back to weighted hashing for now.",
                model_name,
            )
        logger.warning(
            "Inference API embedding failed for '%s', falling back to weighted hashing: %s",
            model_name,
            exc,
        )
        _embedding_mode_cache[model_name] = "hash"
        return _hash_encode([prefixed])[0]


def _encode(
    texts: list[str],
    model_name: str,
) -> list[EmbeddingVector]:
    mode = _embedding_mode_cache.get(model_name, "remote")
    if mode == "hash":
        return _hash_encode(texts)

    try:
        vectors = _feature_extract_encode(texts, model_name)
        _embedding_mode_cache[model_name] = "remote"
        return vectors
    except Exception as exc:
        if "doesn't support task 'feature-extraction'" in str(exc):
            logger.warning(
                "Embedding model '%s' is not compatible with feature_extraction on Hugging Face Inference. "
                "Set EMBEDDING_MODEL to a feature-extraction model such as "
                "'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'. "
                "Falling back to weighted hashing for now.",
                model_name,
            )
        logger.warning(
            "Inference API embedding failed for '%s', falling back to weighted hashing: %s",
            model_name,
            exc,
        )
        _embedding_mode_cache[model_name] = "hash"
        return _hash_encode(texts)
