from __future__ import annotations

import json
from dataclasses import replace

from fastapi.testclient import TestClient

from app.main import create_app
from app.models.schemas import RAGResponse
from app.models.types import GroundingStatus, Language
from app.services.account_rate_limiter import AccountRateLimitDecision
from app.services.rate_limiter import IpRateLimiter
from tests.conftest import make_context


class FakeClock:
    def __init__(self, start: float = 0.0):
        self.value = start

    def now(self) -> float:
        return self.value


class FakeVectorStore:
    index_name = "gapura-rag"

    def get_stats(self):
        return {"total_vector_count": 0}

    def get_index_binding(self):
        return {
            "configured_index": "gapura-rag",
            "active_index": "gapura-rag",
            "embedding_dim": 1024,
            "index_dimension": 1024,
            "metric": "cosine",
        }


def test_chat_sse_returns_grounded_answer_payload(monkeypatch, settings):
    evidence = [
        make_context(
            evidence_id="E1",
            text="The baggage counter opens at 05:00 and closes at 22:00.",
            snippet="The baggage counter opens at 05:00 and closes at 22:00.",
        )
    ]

    def fake_answer_question(**kwargs):
        return RAGResponse(
            answer="The baggage counter opens at 05:00 [E1].",
            detected_language=Language.EN,
            citations=evidence,
            evidence=evidence,
            grounding_status=GroundingStatus.GROUNDED,
            supplement_used=False,
            model_used="fake-model",
        )

    monkeypatch.setattr("app.main.answer_question", fake_answer_question)

    app = create_app(settings=settings, vector_store=FakeVectorStore())
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"question": "When does the baggage counter open?", "language": "en", "history": []},
    )

    events = [
        json.loads(line.removeprefix("data: ").strip())
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    events = [event for event in events if event["type"] != "status"]

    assert events[0] == {
        "type": "token",
        "content": "The baggage counter opens at 05:00 [E1].",
    }
    assert events[1]["type"] == "done"
    assert events[1]["answer"] == "The baggage counter opens at 05:00 [E1]."
    assert events[1]["grounding_status"] == "grounded"
    assert events[1]["evidence"][0]["id"] == "E1"
    assert events[1]["evidence"][0]["snippet"].startswith("The baggage counter")


def test_chat_rate_limit_uses_forwarded_ip_and_returns_retry_after(
    monkeypatch,
    settings,
):
    clock = FakeClock()
    limiter = IpRateLimiter(max_requests=1, window_seconds=10, time_func=clock.now)

    monkeypatch.setattr(
        "app.main.answer_question",
        lambda **kwargs: RAGResponse(
            answer="Supported answer [E1].",
            detected_language=Language.EN,
            citations=[make_context(evidence_id="E1")],
            evidence=[make_context(evidence_id="E1")],
            grounding_status=GroundingStatus.GROUNDED,
            supplement_used=False,
            model_used="fake-model",
        ),
    )

    app = create_app(
        settings=settings,
        vector_store=FakeVectorStore(),
        rate_limiter=limiter,
    )
    client = TestClient(app)
    headers = {"x-forwarded-for": "203.0.113.8, 10.0.0.5"}

    first = client.post(
        "/api/chat",
        headers=headers,
        json={"question": "Question one", "language": "en", "history": []},
    )
    second = client.post(
        "/api/chat",
        headers=headers,
        json={"question": "Question two", "language": "en", "history": []},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["Retry-After"] == "10"
    assert second.json()["retry_after_seconds"] == 10

    clock.value = 11
    third = client.post(
        "/api/chat",
        headers=headers,
        json={"question": "Question three", "language": "en", "history": []},
    )
    assert third.status_code == 200


def test_invalid_question_does_not_consume_rate_limit(monkeypatch, settings):
    clock = FakeClock()
    limiter = IpRateLimiter(max_requests=1, window_seconds=10, time_func=clock.now)

    monkeypatch.setattr(
        "app.main.answer_question",
        lambda **kwargs: RAGResponse(
            answer="Supported answer [E1].",
            detected_language=Language.EN,
            citations=[make_context(evidence_id="E1")],
            evidence=[make_context(evidence_id="E1")],
            grounding_status=GroundingStatus.GROUNDED,
            supplement_used=False,
            model_used="fake-model",
        ),
    )

    app = create_app(
        settings=settings,
        vector_store=FakeVectorStore(),
        rate_limiter=limiter,
    )
    client = TestClient(app)
    headers = {"x-real-ip": "198.51.100.4"}

    bad = client.post(
        "/api/chat",
        headers=headers,
        json={"question": "   ", "language": "en", "history": []},
    )
    good = client.post(
        "/api/chat",
        headers=headers,
        json={"question": "Question one", "language": "en", "history": []},
    )

    assert bad.status_code == 400
    assert good.status_code == 200


def test_account_rate_limit_requires_virtual_assistant_token(monkeypatch, settings):
    app = create_app(
        settings=replace(
            settings,
            account_rate_limit_enabled=True,
            account_rate_limit_consume_url="https://irrs.example/api/consume",
            account_rate_limit_internal_secret="secret",
        ),
        vector_store=FakeVectorStore(),
    )
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        json={"question": "Question one", "language": "en", "history": []},
    )

    assert response.status_code == 401
    assert response.json()["error"] == "Virtual Assistant login is required."


def test_account_rate_limit_uses_bearer_token_before_streaming(
    monkeypatch,
    settings,
):
    seen = {}

    def fake_consume(token, _settings):
        seen["token"] = token
        return AccountRateLimitDecision(allowed=True, remaining=4)

    monkeypatch.setattr("app.main.consume_account_rate_limit", fake_consume)
    monkeypatch.setattr(
        "app.main.answer_question",
        lambda **kwargs: RAGResponse(
            answer="Supported answer [E1].",
            detected_language=Language.EN,
            citations=[make_context(evidence_id="E1")],
            evidence=[make_context(evidence_id="E1")],
            grounding_status=GroundingStatus.GROUNDED,
            supplement_used=False,
            model_used="fake-model",
        ),
    )

    app = create_app(
        settings=replace(
            settings,
            account_rate_limit_enabled=True,
            account_rate_limit_consume_url="https://irrs.example/api/consume",
            account_rate_limit_internal_secret="secret",
        ),
        vector_store=FakeVectorStore(),
    )
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        headers={"Authorization": "Bearer va-token"},
        json={"question": "Question one", "language": "en", "history": []},
    )

    assert response.status_code == 200
    assert seen["token"] == "va-token"
    assert "Supported answer" in response.text


def test_trusted_proxy_secret_skips_account_rate_limit(monkeypatch, settings):
    consumed = False

    def fake_consume(*_args):
        nonlocal consumed
        consumed = True
        return AccountRateLimitDecision(allowed=False, status_code=429)

    monkeypatch.setattr("app.main.consume_account_rate_limit", fake_consume)
    monkeypatch.setattr(
        "app.main.answer_question",
        lambda **kwargs: RAGResponse(
            answer="Proxy answer [E1].",
            detected_language=Language.EN,
            citations=[make_context(evidence_id="E1")],
            evidence=[make_context(evidence_id="E1")],
            grounding_status=GroundingStatus.GROUNDED,
            supplement_used=False,
            model_used="fake-model",
        ),
    )

    app = create_app(
        settings=replace(
            settings,
            account_rate_limit_enabled=True,
            account_rate_limit_consume_url="https://irrs.example/api/consume",
            account_rate_limit_internal_secret="secret",
            trusted_proxy_secret="proxy-secret",
        ),
        vector_store=FakeVectorStore(),
    )
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        headers={"X-Gapura-Proxy-Secret": "proxy-secret"},
        json={"question": "Question one", "language": "en", "history": []},
    )

    assert response.status_code == 200
    assert consumed is False
    assert "Proxy answer" in response.text


def test_account_rate_limit_returns_retry_after(monkeypatch, settings):
    monkeypatch.setattr(
        "app.main.consume_account_rate_limit",
        lambda *_: AccountRateLimitDecision(
            allowed=False,
            status_code=429,
            error="Batas 5 pesan harian untuk Virtual Assistant sudah habis.",
            retry_after_seconds=3600,
            remaining=0,
        ),
    )

    app = create_app(
        settings=replace(
            settings,
            account_rate_limit_enabled=True,
            account_rate_limit_consume_url="https://irrs.example/api/consume",
            account_rate_limit_internal_secret="secret",
        ),
        vector_store=FakeVectorStore(),
    )
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        headers={"Authorization": "Bearer va-token"},
        json={"question": "Question one", "language": "en", "history": []},
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "3600"
    assert response.json()["remaining"] == 0
    assert response.json()["retry_after_seconds"] == 3600


def test_empty_question_does_not_consume_account_rate_limit(monkeypatch, settings):
    consumed = False

    def fake_consume(*_args):
        nonlocal consumed
        consumed = True
        return AccountRateLimitDecision(allowed=True)

    monkeypatch.setattr("app.main.consume_account_rate_limit", fake_consume)

    app = create_app(
        settings=replace(
            settings,
            account_rate_limit_enabled=True,
            account_rate_limit_consume_url="https://irrs.example/api/consume",
            account_rate_limit_internal_secret="secret",
        ),
        vector_store=FakeVectorStore(),
    )
    client = TestClient(app)

    response = client.post(
        "/api/chat",
        headers={"Authorization": "Bearer va-token"},
        json={"question": "   ", "language": "en", "history": []},
    )

    assert response.status_code == 400
    assert consumed is False


def test_stats_exposes_index_binding_metadata(settings):
    app = create_app(settings=settings, vector_store=FakeVectorStore())
    client = TestClient(app)

    response = client.get("/api/stats")

    assert response.status_code == 200
    assert response.json() == {
        "index": "gapura-rag",
        "configured_index": "gapura-rag",
        "active_index": "gapura-rag",
        "total_vectors": 0,
        "embedding_model": settings.embedding_model,
        "embedding_dim": 1024,
        "index_dimension": 1024,
        "pinecone_metric": "cosine",
        "llm_model": settings.llm_model,
    }
