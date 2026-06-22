from __future__ import annotations

from app.models.types import GroundingStatus, Language
from app.services import generator
from tests.conftest import make_context


def test_generate_answer_retries_when_first_response_lacks_inline_citations(
    monkeypatch,
    settings,
    english_context,
):
    responses = iter(
        [
            '{"grounding_status":"grounded","answer":"The counter opens at 05:00.","cited_evidence_ids":["E1"],"supplement":null}',
            '{"grounding_status":"grounded","answer":"The counter opens at 05:00 [E1].","cited_evidence_ids":["E1"],"supplement":null}',
        ]
    )

    monkeypatch.setattr(generator, "_chat_completion", lambda **kwargs: next(responses))

    result = generator.generate_answer(
        question="When does the counter open?",
        contexts=[english_context],
        language=Language.EN,
        settings=settings,
    )

    assert result.grounding_status == GroundingStatus.GROUNDED
    assert "[E1]" in result.answer


def test_generate_answer_appends_server_side_partial_warning(
    monkeypatch,
    settings,
    indonesian_context,
):
    monkeypatch.setattr(
        generator,
        "_chat_completion",
        lambda **kwargs: '{"grounding_status":"partial","answer":"Dokumen menjelaskan jam layanan bagasi [E1].","cited_evidence_ids":["E1"],"supplement":"Di luar dokumen, jam bisa berubah."}',
    )

    result = generator.generate_answer(
        question="Jam layanan bagasi bagaimana?",
        contexts=[indonesian_context],
        language=Language.ID,
        settings=settings,
    )

    assert result.grounding_status == GroundingStatus.PARTIAL
    assert "Peringatan:" in result.answer
    assert "Di luar dokumen" not in result.answer
    assert result.supplement_used is False


def test_generate_answer_keeps_only_cited_contexts_in_citations(
    monkeypatch,
    settings,
):
    contexts = [
        make_context(evidence_id="E1", chunk_id="chunk-1"),
        make_context(
            evidence_id="E2",
            chunk_id="chunk-2",
            chunk_index=1,
            source_filename="manual-2.pdf",
            doc_id="doc-2",
        ),
    ]
    monkeypatch.setattr(
        generator,
        "_chat_completion",
        lambda **kwargs: '{"grounding_status":"grounded","answer":"The supported answer is here [E2].","cited_evidence_ids":["E2"],"supplement":null}',
    )

    result = generator.generate_answer(
        question="What is supported?",
        contexts=contexts,
        language=Language.EN,
        settings=settings,
    )

    assert [ctx.evidence_id for ctx in result.citations] == ["E2"]
    assert [ctx.evidence_id for ctx in result.evidence] == ["E1", "E2"]


def test_generate_answer_accepts_supported_alias_for_grounded(
    monkeypatch,
    settings,
    english_context,
):
    monkeypatch.setattr(
        generator,
        "_chat_completion",
        lambda **kwargs: '{"grounding_status":"supported","answer":"UMNR adalah layanan penumpang anak tanpa pendamping [E1].","cited_evidence_ids":["E1"],"supplement":null}',
    )

    result = generator.generate_answer(
        question="Apa itu UMNR?",
        contexts=[english_context],
        language=Language.ID,
        settings=settings,
    )

    assert result.grounding_status == GroundingStatus.GROUNDED
    assert "[E1]" in result.answer


def test_generate_answer_repairs_missing_inline_citations_from_payload_ids(
    monkeypatch,
    settings,
    english_context,
):
    monkeypatch.setattr(
        generator,
        "_chat_completion",
        lambda **kwargs: '{"grounding_status":"grounded","answer":"UMNR adalah layanan penumpang anak tanpa pendamping.","cited_evidence_ids":["E1"],"supplement":null}',
    )

    result = generator.generate_answer(
        question="Apa itu UMNR?",
        contexts=[english_context],
        language=Language.ID,
        settings=settings,
    )

    assert result.grounding_status == GroundingStatus.GROUNDED
    assert result.answer.endswith("[E1]")
    assert [ctx.evidence_id for ctx in result.citations] == ["E1"]


def test_generate_answer_downgrades_to_unsupported_after_repeated_invalid_output(
    monkeypatch,
    settings,
    english_context,
):
    monkeypatch.setattr(
        generator,
        "_chat_completion",
        lambda **kwargs: '{"grounding_status":"grounded","answer":"The counter opens at 05:00.","cited_evidence_ids":[],"supplement":null}',
    )

    result = generator.generate_answer(
        question="When does the counter open?",
        contexts=[english_context],
        language=Language.EN,
        settings=settings,
    )

    assert result.grounding_status == GroundingStatus.UNSUPPORTED
    assert result.citations == []
    assert result.evidence == []


def test_generate_answer_returns_unsupported_without_contexts(settings):
    result = generator.generate_answer(
        question="What is the weather?",
        contexts=[],
        language=Language.EN,
        settings=settings,
    )

    assert result.grounding_status == GroundingStatus.UNSUPPORTED
    assert result.evidence == []


def test_generate_answer_hides_evidence_when_model_returns_unsupported_with_contexts(
    monkeypatch,
    settings,
    english_context,
):
    monkeypatch.setattr(
        generator,
        "_chat_completion",
        lambda **kwargs: '{"grounding_status":"unsupported","answer":"The document mentions SOP Delay Management.","cited_evidence_ids":[],"supplement":null}',
    )

    result = generator.generate_answer(
        question="Apa saja SOP dalam pelayanan penumpang?",
        contexts=[english_context],
        language=Language.ID,
        settings=settings,
    )

    assert result.grounding_status == GroundingStatus.UNSUPPORTED
    assert result.answer == generator._unsupported_message(Language.ID)
    assert result.citations == []
    assert result.evidence == []


def test_generate_answer_synthesizes_listing_answer_from_sources_when_model_is_unsupported(
    monkeypatch,
    settings,
):
    contexts = [
        make_context(
            evidence_id="E1",
            source_filename="SOP Pelayanan Penumpang.pdf",
            text="Pendahuluan SOP pelayanan penumpang.",
        ),
        make_context(
            evidence_id="E2",
            source_filename="SOP Delay Management.pdf",
            text="Pendahuluan SOP delay management.",
            chunk_id="chunk-2",
            chunk_index=1,
            doc_id="doc-2",
            page=5,
        ),
        make_context(
            evidence_id="E3",
            source_filename="SOP Baggage Irregularity.pdf",
            text="Pendahuluan SOP baggage irregularity.",
            chunk_id="chunk-3",
            chunk_index=2,
            doc_id="doc-3",
            page=6,
        ),
    ]
    monkeypatch.setattr(
        generator,
        "_chat_completion",
        lambda **kwargs: '{"grounding_status":"unsupported","answer":"Tidak ditemukan.","cited_evidence_ids":[],"supplement":null}',
    )

    result = generator.generate_answer(
        question="Apa saja SOP dalam pelayanan penumpang?",
        contexts=contexts,
        language=Language.ID,
        settings=settings,
    )

    assert result.grounding_status == GroundingStatus.PARTIAL
    assert "SOP Pelayanan Penumpang [E1]" in result.answer
    assert "SOP Delay Management [E2]" in result.answer
    assert "SOP Baggage Irregularity [E3]" in result.answer
    assert len(result.citations) == 3
    assert len(result.evidence) == 3


def test_generate_answer_synthesizes_structured_procedure_listing_without_llm(
    monkeypatch,
    settings,
):
    contexts = [
        make_context(
            evidence_id="E1",
            source_filename="SOP Delay Management.pdf",
            text=(
                "3. Flight Delay Handling 3. Penanganan Keterlambatan Penerbangan "
                "Prosedur Penanganan Pesawat Delay."
            ),
        ),
        make_context(
            evidence_id="E2",
            source_filename="SOP Delay Management.pdf",
            text=(
                "4. Passenger Information 4. Informasi Penumpang "
                "Petugas check-in menyampaikan informasi delay."
            ),
            chunk_id="chunk-2",
            chunk_index=1,
            page=4,
        ),
        make_context(
            evidence_id="E3",
            source_filename="SOP Delay Management.pdf",
            text="Preface Foreword Kata Pengantar dokumen ini diterbitkan.",
            chunk_id="chunk-3",
            chunk_index=2,
            page=5,
        ),
    ]

    def should_not_run_llm(**kwargs):
        raise AssertionError("LLM should not be called for structured SOP listing")

    monkeypatch.setattr(generator, "_chat_completion", should_not_run_llm)

    result = generator.generate_answer(
        question="Apa saja SOP dalam penanganan Delay?",
        contexts=contexts,
        language=Language.ID,
        settings=settings,
    )

    assert result.grounding_status == GroundingStatus.PARTIAL
    assert "SOP Delay Management [E1]" in result.answer
    assert "Flight Delay Handling [E1]" in result.answer
    assert "Kata Pengantar" not in result.answer
    assert "Peringatan:" in result.answer
    assert [ctx.evidence_id for ctx in result.citations] == ["E1"]
    assert len(result.evidence) == 3


def test_generate_answer_keeps_standard_unsupported_for_non_listing_questions(
    monkeypatch,
    settings,
    english_context,
):
    monkeypatch.setattr(
        generator,
        "_chat_completion",
        lambda **kwargs: '{"grounding_status":"unsupported","answer":"Tidak ditemukan.","cited_evidence_ids":[],"supplement":null}',
    )

    result = generator.generate_answer(
        question="Apa itu UMNR?",
        contexts=[english_context],
        language=Language.ID,
        settings=settings,
    )

    assert result.grounding_status == GroundingStatus.UNSUPPORTED
    assert result.answer == generator._unsupported_message(Language.ID)
