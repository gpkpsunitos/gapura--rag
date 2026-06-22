from __future__ import annotations

from app.services import pdf_processor


def test_extract_pages_removes_lines_repeated_across_pages():
    page_texts = [
        "STANDARD OPERATING PROCEDURES\nPage 1 of 10\nIsi halaman satu",
        "STANDARD OPERATING PROCEDURES\nPage 2 of 10\nIsi halaman dua",
        "STANDARD OPERATING PROCEDURES\nPage 3 of 10\nIsi halaman tiga",
    ]

    repeated = pdf_processor._detect_repeated_lines(page_texts)

    assert "standard operating procedures" in repeated
    cleaned = pdf_processor._remove_repeated_lines(page_texts[0], repeated)
    assert "STANDARD OPERATING PROCEDURES" not in cleaned
    assert "Isi halaman satu" in cleaned


def test_clean_page_text_dedupes_consecutive_lines_and_controls():
    raw = "Judul\r\nJudul\r\n\uFFFDIsi\t\tutama\r\n\r\n\r\nBaris"

    cleaned = pdf_processor._clean_page_text(raw)

    assert cleaned == "Judul\nIsi utama\nBaris"
