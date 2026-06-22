from __future__ import annotations

import re

from app.models.schemas import DocumentChunk, PageContent
from app.models.types import ChunkId, DocId, Language, build_chunk_id

_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?。])\s+(?=[A-Z\u0041-\u005A\u00C0-\u024F])"
    r"|(?<=\n)\n+"
)

_PARAGRAPH_BOUNDARY = re.compile(r"\n{2,}")
_NON_WORD_RE = re.compile(r"[^A-Za-zÀ-ÿ0-9]+")
_BOILERPLATE_CHUNK_PATTERNS = (
    re.compile(r"\b(preface|foreword|kata pengantar)\b", re.IGNORECASE),
    re.compile(r"\b(lembar persetujuan|internal approval)\b", re.IGNORECASE),
    re.compile(r"\b(effective date|issue\s*-\s*rev|date of issue|revision)\b", re.IGNORECASE),
    re.compile(r"\b(daftar isi|table of contents)\b", re.IGNORECASE),
)


# Complexity: Time O(n) | Space O(n) — n = total characters across all pages
def chunk_pages(
    pages: list[PageContent],
    doc_id: DocId,
    source_filename: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    language: Language = Language.EN,
    parent_retrieval: bool = False,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page in pages:
        if parent_retrieval:
            # Small-to-Big strategy: store leaf chunks and rebuild short context windows at query time.
            sentences = _split_by_sentences(page.text, 400, 50)
            for text in sentences:
                text = _normalize_chunk_text(text)
                if _should_skip_chunk(text):
                    continue
                chunk_id = build_chunk_id(doc_id, page.page_number, chunk_index)
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        text=text,
                        page=page.page_number,
                        chunk_index=chunk_index,
                        language=language,
                        source_filename=source_filename,
                        metadata={"leaf_chunk": True},
                    )
                )
                chunk_index += 1
        else:
            page_chunks = _split_text(page.text, chunk_size, chunk_overlap)
            for text in page_chunks:
                text = _normalize_chunk_text(text)
                if _should_skip_chunk(text):
                    continue
                chunk_id = build_chunk_id(doc_id, page.page_number, chunk_index)
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        text=text,
                        page=page.page_number,
                        chunk_index=chunk_index,
                        language=language,
                        source_filename=source_filename,
                    )
                )
                chunk_index += 1

    return chunks


def _split_text(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    paragraphs = _PARAGRAPH_BOUNDARY.split(text)

    result: list[str] = []
    current_buffer: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        para_len = len(paragraph)

        if current_length + para_len > max_chars and current_buffer:
            merged = "\n\n".join(current_buffer)
            result.append(merged)

            overlap_text = _extract_overlap(merged, overlap_chars)
            current_buffer = [overlap_text] if overlap_text else []
            current_length = len(overlap_text) if overlap_text else 0

        if para_len > max_chars:
            if current_buffer:
                result.append("\n\n".join(current_buffer))
                current_buffer = []
                current_length = 0

            sub_chunks = _split_by_sentences(paragraph, max_chars, overlap_chars)
            result.extend(sub_chunks)
        else:
            current_buffer.append(paragraph)
            current_length += para_len

    if current_buffer:
        result.append("\n\n".join(current_buffer))

    return [c for c in result if c.strip()]


def _split_by_sentences(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    sentences = _SENTENCE_BOUNDARY.split(text)
    if not sentences:
        return _split_by_chars(text, max_chars, overlap_chars)

    result: list[str] = []
    current_buffer: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sent_len = len(sentence)

        if current_length + sent_len > max_chars and current_buffer:
            merged = " ".join(current_buffer)
            result.append(merged)
            overlap_text = _extract_overlap(merged, overlap_chars)
            current_buffer = [overlap_text] if overlap_text else []
            current_length = len(overlap_text) if overlap_text else 0

        if sent_len > max_chars:
            if current_buffer:
                result.append(" ".join(current_buffer))
                current_buffer = []
                current_length = 0
            result.extend(_split_by_chars(sentence, max_chars, overlap_chars))
        else:
            current_buffer.append(sentence)
            current_length += sent_len

    if current_buffer:
        result.append(" ".join(current_buffer))

    return result


def _split_by_chars(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    result: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_chars, text_len)
        result.append(text[start:end])
        start = end - overlap_chars if end < text_len else text_len

    return result


def _extract_overlap(text: str, overlap_chars: int) -> str:
    if len(text) <= overlap_chars:
        return text
    return text[-overlap_chars:]


def _normalize_chunk_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _should_skip_chunk(text: str) -> bool:
    if not text:
        return True
    if len(text) < 40:
        return True

    alpha_count = sum(1 for char in text if char.isalpha())
    if alpha_count < 20:
        return True

    normalized = _NON_WORD_RE.sub(" ", text.lower()).strip()
    unique_tokens = {token for token in normalized.split() if token}
    if len(unique_tokens) < 6:
        return True

    if any(pattern.search(text) for pattern in _BOILERPLATE_CHUNK_PATTERNS):
        dense_keywords = {"delay", "penumpang", "baggage", "ground", "flight", "handling"}
        if len(unique_tokens & dense_keywords) < 2:
            return True

    return False
