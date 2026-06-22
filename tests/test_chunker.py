from __future__ import annotations

from app.models.schemas import PageContent
from app.models.types import DocId, Language
from app.pipelines import ingest
from app.services import chunker


def test_chunk_pages_skips_low_value_boilerplate_chunks():
    pages = [
        PageContent(
            page_number=1,
            text=(
                "Preface Foreword Kata Pengantar\n"
                "Date of Issue : 10 Sep 2024\n"
                "Issue - Rev : 02 - 00"
            ),
        ),
        PageContent(
            page_number=2,
            text=(
                "Flight Delay Handling\n\n"
                "Petugas check-in menyampaikan informasi delay kepada penumpang "
                "dan melakukan koordinasi dengan unit terkait."
            ),
        ),
    ]

    chunks = chunker.chunk_pages(
        pages=pages,
        doc_id=DocId("doc-1"),
        source_filename="SOP Delay Management.pdf",
        chunk_size=512,
        chunk_overlap=64,
        language=Language.ID,
        parent_retrieval=False,
    )

    assert len(chunks) == 1
    assert "Preface" not in chunks[0].text
    assert "informasi delay" in chunks[0].text


def test_embedding_text_enriches_chunk_with_source_and_page():
    page = PageContent(
        page_number=7,
        text="Petugas ramp berkoordinasi dengan unit terkait.",
    )
    chunk = chunker.chunk_pages(
        pages=[page],
        doc_id=DocId("doc-1"),
        source_filename="SOP Delay Management.pdf",
        language=Language.ID,
        parent_retrieval=False,
    )[0]

    enriched = ingest._embedding_text(chunk)

    assert "Source: SOP Delay Management" in enriched
    assert "Page: 7" in enriched
    assert "Petugas ramp berkoordinasi" in enriched
