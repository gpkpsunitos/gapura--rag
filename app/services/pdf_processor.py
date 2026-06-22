from __future__ import annotations

import io
import re
from typing import BinaryIO

from app.models.schemas import PageContent

_HEADER_FOOTER_PATTERN = re.compile(
    r"^(?:page\s*\d+|halaman\s*\d+|\d+\s*(?:of|dari)\s*\d+).*$",
    re.IGNORECASE | re.MULTILINE,
)
_WHITESPACE_RE = re.compile(r"[ \t]+")
_CONTROL_CHAR_RE = re.compile(r"[\u0000-\u0008\u000B-\u001F\u007F]+")


# Complexity: Time O(p) | Space O(p) — p = page count
def extract_pages(file_bytes: bytes, filename: str) -> list[PageContent]:
    stream = io.BytesIO(file_bytes)
    return _extract_from_stream(stream)


def _extract_from_stream(stream: BinaryIO) -> list[PageContent]:
    import pdfplumber

    raw_pages: list[tuple[int, str]] = []
    with pdfplumber.open(stream) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned = _clean_page_text(raw_text)
            if cleaned:
                raw_pages.append((page_num, cleaned))

    repeated_lines = _detect_repeated_lines([text for _, text in raw_pages])
    pages: list[PageContent] = []
    for page_num, text in raw_pages:
        cleaned = _remove_repeated_lines(text, repeated_lines)
        if cleaned.strip():
            pages.append(PageContent(page_number=page_num, text=cleaned))
    return pages


def _clean_page_text(text: str) -> str:
    text = text.replace("\uFFFD", " ")
    text = _CONTROL_CHAR_RE.sub(" ", text)
    text = _HEADER_FOOTER_PATTERN.sub("", text)
    text = text.replace("\r", "\n")
    lines = [_normalize_line(line) for line in text.splitlines()]
    deduped_lines: list[str] = []
    last_line = ""
    for line in lines:
        if not line:
            continue
        if line == last_line:
            continue
        deduped_lines.append(line)
        last_line = line
    text = "\n".join(deduped_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_line(line: str) -> str:
    line = _WHITESPACE_RE.sub(" ", line)
    return line.strip()


def _line_key(line: str) -> str:
    normalized = line.lower().strip()
    normalized = re.sub(r"\b\d+\b", "#", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _detect_repeated_lines(page_texts: list[str]) -> set[str]:
    counts: dict[str, int] = {}
    for text in page_texts:
        seen_on_page: set[str] = set()
        for line in text.splitlines():
            normalized = _line_key(line)
            if len(normalized) < 8:
                continue
            if normalized in seen_on_page:
                continue
            seen_on_page.add(normalized)
            counts[normalized] = counts.get(normalized, 0) + 1

    repeated: set[str] = set()
    for line_key, count in counts.items():
        if count >= 3:
            repeated.add(line_key)
    return repeated


def _remove_repeated_lines(text: str, repeated_lines: set[str]) -> str:
    kept_lines = [
        line
        for line in text.splitlines()
        if _line_key(line) not in repeated_lines
    ]
    return "\n".join(kept_lines).strip()
