from __future__ import annotations

import numpy as np

from app.services import embedder


def test_embed_query_uses_feature_extraction_when_available(monkeypatch):
    embedder._embedding_mode_cache.clear()
    embedder._feature_extract_encode_single_cached.cache_clear()

    class FakeClient:
        def feature_extraction(self, text, model=None, normalize=None):
            assert text.startswith("query: ")
            assert model == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            assert normalize is True
            return np.array([0.6, 0.8], dtype=np.float32)

    monkeypatch.setattr(embedder, "_get_client", lambda: FakeClient())

    vector = embedder.embed_query(
        "tujuan peraturan pm 89",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )

    assert len(vector) == 2
    assert round(vector[0], 3) == 0.6
    assert (
        embedder._embedding_mode_cache["sentence-transformers/paraphrase-multilingual-mpnet-base-v2"]
        == "remote"
    )


def test_embed_query_falls_back_to_hash_embeddings_when_remote_fails(monkeypatch):
    embedder._embedding_mode_cache.clear()
    embedder._feature_extract_encode_single_cached.cache_clear()

    class FakeClient:
        def feature_extraction(self, text, model=None, normalize=None):
            raise RuntimeError("remote failed")

    monkeypatch.setattr(embedder, "_get_client", lambda: FakeClient())

    monkeypatch.setattr(
        embedder,
        "_hash_encode",
        lambda texts, dim=768: [[1.0] + [0.0] * (dim - 1) for _ in texts],
    )

    vector = embedder.embed_query(
        "tujuan peraturan pm 89",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )

    assert len(vector) == 768
    assert vector[0] == 1.0
    assert (
        embedder._embedding_mode_cache["sentence-transformers/paraphrase-multilingual-mpnet-base-v2"]
        == "hash"
    )


def test_embed_query_uses_cache_for_repeated_queries(monkeypatch):
    embedder._embedding_mode_cache.clear()
    embedder._feature_extract_encode_single_cached.cache_clear()
    calls = {"count": 0}

    class FakeClient:
        def feature_extraction(self, text, model=None, normalize=None):
            calls["count"] += 1
            return np.array([0.6, 0.8], dtype=np.float32)

    monkeypatch.setattr(embedder, "_get_client", lambda: FakeClient())

    vector_one = embedder.embed_query(
        "tujuan peraturan pm 89",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    vector_two = embedder.embed_query(
        "tujuan peraturan pm 89",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )

    assert vector_one == vector_two
    assert calls["count"] == 1
