from __future__ import annotations

import pytest

from app.config import Settings
from app.models.schemas import RetrievedContext
from app.models.types import ChunkId, DocId, Language


@pytest.fixture
def settings() -> Settings:
    return Settings(
        hf_token="test-token",
        pinecone_api_key="test-pinecone",
        hyde_enabled=False,
        rerank_enabled=False,
        context_window_radius=0,
        retrieval_min_score=0.3,
        min_supporting_evidence=1,
    )


def make_context(
    *,
    evidence_id: str = "",
    text: str = "Document support text with enough detail to pass retrieval threshold.",
    snippet: str | None = None,
    score: float = 0.9,
    rerank_score: float | None = None,
    source_filename: str = "manual.pdf",
    page: int = 2,
    chunk_id: str = "chunk-1",
    doc_id: str = "doc-1",
    chunk_index: int = 0,
) -> RetrievedContext:
    return RetrievedContext(
        evidence_id=evidence_id,
        text=text,
        snippet=snippet or text,
        score=score,
        rerank_score=rerank_score,
        source_filename=source_filename,
        page=page,
        chunk_id=ChunkId(chunk_id),
        doc_id=DocId(doc_id),
        chunk_index=chunk_index,
    )


@pytest.fixture
def english_context() -> RetrievedContext:
    return make_context(
        evidence_id="E1",
        text="The baggage counter opens at 05:00 and closes at 22:00.",
        snippet="The baggage counter opens at 05:00 and closes at 22:00.",
    )


@pytest.fixture
def indonesian_context() -> RetrievedContext:
    return make_context(
        evidence_id="E1",
        text="Layanan bagasi dibuka pukul 05.00 dan ditutup pukul 22.00.",
        snippet="Layanan bagasi dibuka pukul 05.00 dan ditutup pukul 22.00.",
        source_filename="panduan.pdf",
        page=3,
        chunk_id="chunk-id",
        doc_id="doc-id",
        chunk_index=1,
    )


__all__ = ["Language", "make_context"]
