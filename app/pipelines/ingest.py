from __future__ import annotations

import logging

from app.config import Settings
from app.models.schemas import DocumentChunk, IngestionResult
from app.models.types import compute_doc_id
from app.services.chunker import chunk_pages
from app.services.embedder import embed_passages
from app.services.language import detect_language
from app.services.pdf_processor import extract_pages
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


# Complexity: Time O(p + n + n/b) | Space O(n × d)
# p=pages, n=chunks, b=batch_size, d=embedding_dim
def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    settings: Settings,
    vector_store: VectorStore,
    replace_existing: bool = False,
) -> IngestionResult:
    doc_id = compute_doc_id(file_bytes)

    if vector_store.doc_exists(doc_id):
        if replace_existing:
            logger.info("Re-ingesting existing document: %s (%s)", filename, doc_id)
            vector_store.delete_by_doc_id(doc_id)
        else:
            logger.info("Document already ingested: %s (%s)", filename, doc_id)
            return IngestionResult(
                doc_id=doc_id,
                source_filename=filename,
                total_pages=0,
                total_chunks=0,
                skipped=True,
            )

    pages = extract_pages(file_bytes, filename)
    if not pages:
        raise ValueError(f"No extractable text found in PDF: {filename}")

    sample_text = " ".join(p.text[:200] for p in pages[:3])
    doc_language = detect_language(sample_text, settings.language_confidence_threshold)

    chunks = chunk_pages(
        pages=pages,
        doc_id=doc_id,
        source_filename=filename,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        language=doc_language,
        parent_retrieval=settings.parent_retrieval_enabled,
    )

    per_chunk_languages = [
        detect_language(c.text, settings.language_confidence_threshold)
        for c in chunks
    ]
    for chunk, lang in zip(chunks, per_chunk_languages):
        chunk.language = lang

    texts = [_embedding_text(c) for c in chunks]
    embeddings = embed_passages(texts, settings.embedding_model)

    for chunk, emb in zip(chunks, embeddings):
        chunk.embedding = list(emb)

    vector_store.upsert_chunks(chunks)

    logger.info(
        "Ingested %s: %d pages, %d chunks, language=%s",
        filename,
        len(pages),
        len(chunks),
        doc_language.value,
    )

    return IngestionResult(
        doc_id=doc_id,
        source_filename=filename,
        total_pages=len(pages),
        total_chunks=len(chunks),
    )


def _embedding_text(chunk: DocumentChunk) -> str:
    source_title = chunk.source_filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
    source_title = " ".join(source_title.split()).strip()
    return (
        f"Source: {source_title}\n"
        f"Page: {chunk.page}\n"
        f"Content: {chunk.text}"
    )
