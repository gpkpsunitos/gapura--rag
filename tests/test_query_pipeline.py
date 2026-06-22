from __future__ import annotations

from app.models.schemas import RAGResponse
from app.models.types import GroundingStatus, Language
from app.pipelines import query


def test_answer_question_refuses_when_retrieval_returns_no_evidence(monkeypatch, settings):
    monkeypatch.setattr(query, "detect_language", lambda *_: Language.EN)
    monkeypatch.setattr(query, "reformulate_query", lambda *args, **kwargs: "What is the policy?")
    monkeypatch.setattr(query, "retrieve", lambda **kwargs: [])

    response = query.answer_question(
        question="What is the policy?",
        settings=settings,
        vector_store=object(),
    )

    assert response.grounding_status == GroundingStatus.UNSUPPORTED
    assert response.evidence == []


def test_answer_question_uses_reformulated_follow_up(monkeypatch, settings, english_context):
    seen = {}

    monkeypatch.setattr(query, "detect_language", lambda *_: Language.EN)

    def fake_reformulate(message, history, _settings):
        seen["history"] = history
        return "reformulated question"

    def fake_retrieve(**kwargs):
        seen["query"] = kwargs["query"]
        return [english_context]

    def fake_generate(question, contexts, language, _settings, history=None):
        return RAGResponse(
            answer="Supported answer [E1].",
            detected_language=language,
            citations=contexts,
            evidence=contexts,
            grounding_status=GroundingStatus.GROUNDED,
            supplement_used=False,
            model_used="fake-model",
        )

    monkeypatch.setattr(query, "reformulate_query", fake_reformulate)
    monkeypatch.setattr(query, "retrieve", fake_retrieve)
    monkeypatch.setattr(query, "generate_answer", fake_generate)

    history = [{"role": "user", "content": "What about baggage?"}]
    response = query.answer_question(
        question="What about that?",
        settings=settings,
        vector_store=object(),
        history=history,
    )

    assert seen["history"] == history
    assert seen["query"] == "reformulated question"
    assert response.grounding_status == GroundingStatus.GROUNDED
