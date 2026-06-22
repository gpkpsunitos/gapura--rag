from __future__ import annotations

import pytest
from app.config import Settings
from app.models.types import Language, DocId, ChunkId
from app.services import retriever
from tests.conftest import make_context

class MockVectorStore:
    def __init__(self):
        self.last_query_sources = None

    def query_similar(self, embedding, top_k, filter_dict=None, sources=None):
        self.last_query_sources = sources
        # Simulate results from different sources
        return [
            make_context(chunk_id="a1", source_filename="a.pdf"),
            make_context(chunk_id="b1", source_filename="b.pdf"),
            make_context(chunk_id="c1", source_filename="c.pdf"),
        ]

    def expand_contexts(self, contexts, radius):
        return contexts

def test_retrieve_passes_sources_to_vector_store(monkeypatch):
    settings = Settings(
        hf_token="token",
        pinecone_api_key="pinecone",
        hyde_enabled=False,
        rerank_enabled=False,
    )
    store = MockVectorStore()
    
    monkeypatch.setattr(retriever, "embed_query", lambda text, _: [0.1] * 768)

    # Test with specific sources
    test_sources = ["a.pdf", "b.pdf"]
    retriever.retrieve(
        query="test",
        settings=settings,
        vector_store=store,
        language=Language.EN,
        sources=test_sources
    )
    
    assert store.last_query_sources == test_sources

    # Test without sources
    retriever.retrieve(
        query="test",
        settings=settings,
        vector_store=store,
        language=Language.EN,
    )
    
    assert store.last_query_sources is None
