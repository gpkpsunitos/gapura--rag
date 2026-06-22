from __future__ import annotations

from app.models.schemas import PageContent
from app.models.types import Language
from app.models.types import DocId
from app.pipelines import ingest


class FakeVectorStore:
    def __init__(self, exists: bool) -> None:
        self.exists = exists
        self.deleted: list[DocId] = []
        self.upserted_chunks = 0

    def doc_exists(self, doc_id: DocId) -> bool:
        return self.exists

    def delete_by_doc_id(self, doc_id: DocId) -> None:
        self.deleted.append(doc_id)

    def upsert_chunks(self, chunks) -> int:
        self.upserted_chunks = len(chunks)
        return len(chunks)


def test_ingest_pdf_skips_existing_document_by_default(monkeypatch, settings):
    vector_store = FakeVectorStore(exists=True)

    monkeypatch.setattr(
        ingest,
        "extract_pages",
        lambda file_bytes, filename: [PageContent(page_number=1, text="Isi dokumen.")],
    )

    result = ingest.ingest_pdf(
        b"same-file",
        "manual.pdf",
        settings,
        vector_store,
    )

    assert result.skipped is True
    assert vector_store.deleted == []
    assert vector_store.upserted_chunks == 0


def test_ingest_pdf_replaces_existing_document_when_requested(monkeypatch, settings):
    vector_store = FakeVectorStore(exists=True)

    monkeypatch.setattr(
        ingest,
        "extract_pages",
        lambda file_bytes, filename: [
            PageContent(
                page_number=1,
                text="Flight delay handling untuk penumpang dan koordinasi operasional.",
            )
        ],
    )
    monkeypatch.setattr(
        ingest,
        "detect_language",
        lambda text, threshold=0.5: Language.ID,
    )
    monkeypatch.setattr(
        ingest,
        "embed_passages",
        lambda texts, model_name: [[0.1] * settings.embedding_dim for _ in texts],
    )

    result = ingest.ingest_pdf(
        b"same-file",
        "manual.pdf",
        settings,
        vector_store,
        replace_existing=True,
    )

    assert result.skipped is False
    assert len(vector_store.deleted) == 1
    assert vector_store.upserted_chunks > 0
