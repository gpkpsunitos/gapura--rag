from __future__ import annotations

import pytest

from app.config import Settings
from app.services.vector_store import VectorStore


class FakeIndex:
    def __init__(self, name: str):
        self.name = name

    def describe_index_stats(self):
        return {"total_vector_count": 42}


class FakePineconeClient:
    def __init__(self, indexes):
        self._indexes = list(indexes)
        self.created = []
        self.index_requests = []

    def list_indexes(self):
        return list(self._indexes)

    def create_index(self, name, dimension, metric, spec):
        del metric, spec
        self.created.append((name, dimension))
        self._indexes.append({"name": name, "dimension": dimension})

    def Index(self, name):
        self.index_requests.append(name)
        return FakeIndex(name)


def test_ensure_index_fails_fast_when_dimension_mismatches():
    settings = Settings(
        hf_token="token",
        pinecone_api_key="pinecone",
        pinecone_index="gapura-rag",
        embedding_model="intfloat/multilingual-e5-large",
        embedding_dim=1024,
    )
    store = VectorStore(settings)
    store._client = FakePineconeClient([{"name": "gapura-rag", "dimension": 768}])

    with pytest.raises(ValueError, match="has dimension 768"):
        store.ensure_index()

    assert store._client.created == []


def test_ensure_index_keeps_configured_index_when_dimension_matches():
    settings = Settings(
        hf_token="token",
        pinecone_api_key="pinecone",
        pinecone_index="gapura-rag",
        embedding_dim=768,
    )
    store = VectorStore(settings)
    store._client = FakePineconeClient([{"name": "gapura-rag", "dimension": 768}])

    store.ensure_index()

    assert store.index_name == "gapura-rag"
    assert store.index_dimension == 768
    assert store._client.created == []
    assert store._client.index_requests[-1] == "gapura-rag"


def test_get_index_binding_reports_active_index_metadata():
    settings = Settings(
        hf_token="token",
        pinecone_api_key="pinecone",
        pinecone_index="gapura-rag",
        embedding_dim=768,
        pinecone_metric="cosine",
    )
    store = VectorStore(settings)
    store._client = FakePineconeClient([{"name": "gapura-rag", "dimension": 768}])

    store.ensure_index()

    assert store.get_index_binding() == {
        "configured_index": "gapura-rag",
        "active_index": "gapura-rag",
        "embedding_dim": 768,
        "index_dimension": 768,
        "metric": "cosine",
    }
