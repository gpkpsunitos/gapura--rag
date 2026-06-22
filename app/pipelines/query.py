from __future__ import annotations

import logging
from typing import Generator

from app.config import Settings
from app.models.schemas import RAGResponse, RetrievedContext
from app.models.types import Language
from app.services.generator import (
    generate_answer,
    generate_answer_plain_stream,
    generate_answer_stream,
    generate_chitchat_answer,
    reformulate_query,
)
from app.services.language import detect_language
from app.services.retriever import retrieve
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

_CHITCHAT_MARKER = "CHITCHAT"


# Complexity: Time O(k × s) retrieval + O(1) LLM call | Space O(k + t)
def answer_question(
    question: str,
    settings: Settings,
    vector_store: VectorStore,
    language_override: Language | None = None,
    top_k: int | None = None,
    history: list[dict[str, str]] | None = None,
    sources: list[str] | None = None,
) -> RAGResponse:
    language = language_override or detect_language(
        question, settings.language_confidence_threshold
    )

    search_query = reformulate_query(question, history, settings)

    if search_query.strip().upper() == _CHITCHAT_MARKER:
        return generate_chitchat_answer(question, language, settings, history)

    contexts = retrieve(
        query=search_query,
        settings=settings,
        vector_store=vector_store,
        language=language,
        top_k=top_k,
        sources=sources,
    )

    return generate_answer(question, contexts, language, settings, history)


# Complexity: Time O(k × s) retrieval + O(1) streaming LLM call | Space O(k)
def answer_question_stream(
    question: str,
    settings: Settings,
    vector_store: VectorStore,
    language_override: Language | None = None,
    top_k: int | None = None,
    history: list[dict[str, str]] | None = None,
    sources: list[str] | None = None,
) -> tuple[Language, list[RetrievedContext], Generator[str, None, None]]:
    language = language_override or detect_language(
        question, settings.language_confidence_threshold
    )

    search_query = reformulate_query(question, history, settings)

    if search_query.strip().upper() == _CHITCHAT_MARKER:
        response = generate_chitchat_answer(question, language, settings, history)
        return language, response.evidence, iter([response.answer])

    contexts = retrieve(
        query=search_query,
        settings=settings,
        vector_store=vector_store,
        language=language,
        top_k=top_k,
    )

    token_stream = generate_answer_stream(question, contexts, language, settings, history)
    return language, contexts, token_stream


def answer_question_plain_stream(
    question: str,
    settings: Settings,
    vector_store: VectorStore,
    language_override: Language | None = None,
    top_k: int | None = None,
    history: list[dict[str, str]] | None = None,
    sources: list[str] | None = None,
) -> tuple[Language, list[RetrievedContext], Generator[str, None, None]]:
    language = language_override or detect_language(
        question, settings.language_confidence_threshold
    )

    search_query = reformulate_query(question, history, settings)

    if search_query.strip().upper() == _CHITCHAT_MARKER:
        response = generate_chitchat_answer(question, language, settings, history)
        return language, response.evidence, iter([response.answer])

    contexts = retrieve(
        query=search_query,
        settings=settings,
        vector_store=vector_store,
        language=language,
        top_k=top_k,
        sources=sources,
        expand_contexts=False,
    )

    token_stream = generate_answer_plain_stream(
        question,
        contexts,
        language,
        settings,
        history,
    )
    return language, contexts, token_stream
