from __future__ import annotations

from app.config import Settings
from app.models.types import Language
from app.services import retriever
from tests.conftest import make_context


class FakeVectorStore:
    def __init__(self, responses):
        self.responses = responses
        self.expand_calls = []
        self.query_calls = []

    def query_similar(self, embedding, top_k, filter_dict=None, sources=None):
        key = embedding[0]
        self.query_calls.append(
            {
                "query": key,
                "top_k": top_k,
                "filter_dict": filter_dict,
                "sources": sources,
            }
        )
        return [ctx.model_copy() for ctx in self.responses.get(key, [])][:top_k]

    def expand_contexts(self, contexts, radius):
        self.expand_calls.append((len(contexts), radius))
        return contexts


def test_retrieve_uses_hyde_only_after_weak_primary(monkeypatch):
    settings = Settings(
        hf_token="token",
        pinecone_api_key="pinecone",
        hyde_enabled=True,
        multi_query_enabled=False,
        rerank_enabled=False,
        context_window_radius=0,
        retrieval_min_score=0.5,
        min_supporting_evidence=1,
    )
    store = FakeVectorStore(
        {
            "original": [make_context(score=0.2)],
            "hyde": [make_context(score=0.91)],
        }
    )

    monkeypatch.setattr(retriever, "embed_query", lambda text, _: [text])
    monkeypatch.setattr(retriever, "generate_hypothetical_answer", lambda query, _: "hyde")

    contexts = retriever.retrieve(
        query="original",
        settings=settings,
        vector_store=store,
        language=Language.ID,
        top_k=1,
    )

    assert len(contexts) == 1
    assert contexts[0].evidence_id == "E1"
    assert contexts[0].score == 0.91
    assert [call["query"] for call in store.query_calls] == ["original", "hyde"]


def test_retrieve_stops_after_strong_primary_before_expansions(monkeypatch):
    settings = Settings(
        hf_token="token",
        pinecone_api_key="pinecone",
        hyde_enabled=True,
        multi_query_enabled=True,
        rerank_enabled=False,
        context_window_radius=0,
        retrieval_min_score=0.5,
        min_supporting_evidence=1,
    )
    store = FakeVectorStore({"question": [make_context(score=0.91)]})
    variation_calls = {"count": 0}
    hyde_calls = {"count": 0}

    monkeypatch.setattr(retriever, "embed_query", lambda text, _: [text])

    def fake_variations(*args, **kwargs):
        variation_calls["count"] += 1
        return ["variation"]

    def fake_hyde(*args, **kwargs):
        hyde_calls["count"] += 1
        return "hyde"

    monkeypatch.setattr(retriever, "generate_query_variations", fake_variations)
    monkeypatch.setattr(retriever, "generate_hypothetical_answer", fake_hyde)

    contexts = retriever.retrieve(
        query="question",
        settings=settings,
        vector_store=store,
        language=Language.EN,
        top_k=1,
    )

    assert len(contexts) == 1
    assert variation_calls["count"] == 0
    assert hyde_calls["count"] == 0
    assert [call["query"] for call in store.query_calls] == ["question"]


def test_retrieve_honors_top_k_after_reranking(monkeypatch):
    settings = Settings(
        hf_token="token",
        pinecone_api_key="pinecone",
        hyde_enabled=False,
        multi_query_enabled=False,
        rerank_enabled=True,
        rerank_top_n=1,
        context_window_radius=0,
        retrieval_min_score=0.1,
        rerank_min_score=0.0,
        min_supporting_evidence=1,
    )
    store = FakeVectorStore(
        {
            "question": [
                make_context(chunk_id="c1", chunk_index=0, score=0.9),
                make_context(chunk_id="c2", chunk_index=1, score=0.85),
                make_context(chunk_id="c3", chunk_index=2, score=0.83),
            ]
        }
    )

    monkeypatch.setattr(retriever, "embed_query", lambda text, _: [text])

    contexts = retriever.retrieve(
        query="question",
        settings=settings,
        vector_store=store,
        language=Language.EN,
        top_k=2,
    )

    assert len(contexts) == 2
    assert contexts[0].chunk_id == "c1"
    assert contexts[1].chunk_id == "c2"
    assert contexts[0].rerank_score is not None
    assert contexts[1].rerank_score is not None


def test_retrieve_skips_english_reranker_for_indonesian(monkeypatch):
    settings = Settings(
        hf_token="token",
        pinecone_api_key="pinecone",
        hyde_enabled=False,
        multi_query_enabled=False,
        rerank_enabled=True,
        context_window_radius=0,
        min_supporting_evidence=1,
    )
    store = FakeVectorStore({"pertanyaan": [make_context(score=0.8)]})

    monkeypatch.setattr(retriever, "embed_query", lambda text, _: [text])

    contexts = retriever.retrieve(
        query="pertanyaan",
        settings=settings,
        vector_store=store,
        language=Language.ID,
        top_k=1,
    )

    assert len(contexts) == 1
    assert contexts[0].rerank_score is not None


def test_retrieve_lexical_rerank_prefers_overlap(monkeypatch):
    settings = Settings(
        hf_token="token",
        pinecone_api_key="pinecone",
        hyde_enabled=False,
        multi_query_enabled=False,
        rerank_enabled=True,
        context_window_radius=0,
        retrieval_min_score=0.1,
        min_supporting_evidence=1,
    )
    store = FakeVectorStore(
        {
            "question": [
                make_context(
                    text="question exact match with enough detail to pass the retrieval length threshold",
                    snippet="question exact match with enough detail to pass the retrieval length threshold",
                    score=0.4,
                ),
                make_context(
                    chunk_id="c2",
                    chunk_index=1,
                    text="different words entirely with enough detail to pass the retrieval length threshold",
                    snippet="different words entirely with enough detail to pass the retrieval length threshold",
                    score=0.9,
                ),
            ]
        }
    )

    monkeypatch.setattr(retriever, "embed_query", lambda text, _: [text])

    contexts = retriever.retrieve(
        query="question",
        settings=settings,
        vector_store=store,
        language=Language.EN,
        top_k=1,
    )

    assert len(contexts) == 1
    assert contexts[0].chunk_id == "chunk-1"


def test_retrieve_accepts_context_when_rerank_is_strong_even_if_dense_score_is_low(monkeypatch):
    settings = Settings(
        hf_token="token",
        pinecone_api_key="pinecone",
        hyde_enabled=False,
        multi_query_enabled=False,
        rerank_enabled=True,
        context_window_radius=0,
        retrieval_min_score=0.9,
        rerank_min_score=0.2,
        min_supporting_evidence=1,
    )
    store = FakeVectorStore(
        {
            "kategori keterlambatan penerbangan": [
                make_context(
                    text="kategori keterlambatan penerbangan dengan detail yang cukup untuk melewati threshold retrieval",
                    snippet="kategori keterlambatan penerbangan dengan detail yang cukup untuk melewati threshold retrieval",
                    score=0.05,
                )
            ]
        }
    )

    monkeypatch.setattr(retriever, "embed_query", lambda text, _: [text])

    contexts = retriever.retrieve(
        query="kategori keterlambatan penerbangan",
        settings=settings,
        vector_store=store,
        language=Language.ID,
        top_k=1,
    )

    assert len(contexts) == 1
    assert contexts[0].rerank_score is not None


def test_retrieve_keeps_dense_order_when_rerank_disabled(monkeypatch):
    settings = Settings(
        hf_token="token",
        pinecone_api_key="pinecone",
        hyde_enabled=False,
        multi_query_enabled=False,
        rerank_enabled=False,
        context_window_radius=0,
        retrieval_min_score=0.1,
        min_supporting_evidence=1,
    )
    store = FakeVectorStore(
        {
            "question": [
                make_context(
                    text="question exact match with enough detail to pass the retrieval length threshold",
                    snippet="question exact match with enough detail to pass the retrieval length threshold",
                    score=0.4,
                ),
                make_context(
                    chunk_id="c2",
                    chunk_index=1,
                    text="different words entirely with enough detail to pass the retrieval length threshold",
                    snippet="different words entirely with enough detail to pass the retrieval length threshold",
                    score=0.9,
                ),
            ]
        }
    )

    monkeypatch.setattr(retriever, "embed_query", lambda text, _: [text])

    contexts = retriever.retrieve(
        query="question",
        settings=settings,
        vector_store=store,
        language=Language.EN,
        top_k=1,
    )

    assert len(contexts) == 1
    assert contexts[0].chunk_id == "c2"
    assert contexts[0].rerank_score is None


def test_retrieve_spreads_contexts_across_sources_when_available(monkeypatch):
    settings = Settings(
        hf_token="token",
        pinecone_api_key="pinecone",
        hyde_enabled=False,
        multi_query_enabled=False,
        rerank_enabled=False,
        context_window_radius=0,
        retrieval_min_score=0.1,
        source_diversity_cap=2,
        min_supporting_evidence=1,
    )
    store = FakeVectorStore(
        {
            "question": [
                make_context(chunk_id="a1", score=0.99, source_filename="source-a.pdf"),
                make_context(chunk_id="a2", chunk_index=1, score=0.98, source_filename="source-a.pdf"),
                make_context(chunk_id="a3", chunk_index=2, score=0.97, source_filename="source-a.pdf"),
                make_context(
                    chunk_id="b1",
                    chunk_index=3,
                    score=0.96,
                    source_filename="source-b.pdf",
                    doc_id="doc-b",
                ),
                make_context(
                    chunk_id="b2",
                    chunk_index=4,
                    score=0.95,
                    source_filename="source-b.pdf",
                    doc_id="doc-b",
                ),
            ]
        }
    )

    monkeypatch.setattr(retriever, "embed_query", lambda text, _: [text])

    contexts = retriever.retrieve(
        query="question",
        settings=settings,
        vector_store=store,
        language=Language.EN,
        top_k=4,
    )

    assert len(contexts) == 4
    assert {ctx.source_filename for ctx in contexts} == {"source-a.pdf", "source-b.pdf"}
    assert [ctx.source_filename for ctx in contexts].count("source-a.pdf") == 2
    assert [ctx.source_filename for ctx in contexts].count("source-b.pdf") == 2


def test_retrieve_rerank_uses_source_filename_overlap(monkeypatch):
    settings = Settings(
        hf_token="token",
        pinecone_api_key="pinecone",
        hyde_enabled=False,
        multi_query_enabled=False,
        rerank_enabled=True,
        context_window_radius=0,
        retrieval_min_score=0.1,
        rerank_min_score=0.0,
        min_supporting_evidence=1,
    )
    store = FakeVectorStore(
        {
            "Apa tujuan utama GOM": [
                make_context(
                    chunk_id="pm89",
                    score=0.95,
                    text="dokumen operasional bandara dengan detail yang cukup untuk melewati threshold retrieval",
                    snippet="dokumen operasional bandara dengan detail yang cukup untuk melewati threshold retrieval",
                    source_filename="PM_89_TAHUN_2015.pdf",
                    doc_id="pm89",
                ),
                make_context(
                    chunk_id="gom",
                    chunk_index=1,
                    score=0.8,
                    text="dokumen operasional bandara dengan detail yang cukup untuk melewati threshold retrieval",
                    snippet="dokumen operasional bandara dengan detail yang cukup untuk melewati threshold retrieval",
                    source_filename="GOM OP 01 Rev 10.pdf",
                    doc_id="gom",
                ),
            ]
        }
    )

    monkeypatch.setattr(retriever, "embed_query", lambda text, _: [text])

    contexts = retriever.retrieve(
        query="Apa tujuan utama GOM",
        settings=settings,
        vector_store=store,
        language=Language.ID,
        top_k=1,
    )

    assert len(contexts) == 1
    assert contexts[0].source_filename == "GOM OP 01 Rev 10.pdf"
