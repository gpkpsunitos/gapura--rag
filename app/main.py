from __future__ import annotations

import json
import importlib.util
import hmac
import logging
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError

from app.config import Settings, load_settings
from app.models.schemas import RAGResponse, RetrievedContext
from app.models.types import Language, compute_doc_id
from app.pipelines.ingest import ingest_pdf
from app.pipelines.query import answer_question, answer_question_plain_stream
from app.services.account_rate_limiter import (
    consume_account_rate_limit,
    extract_bearer_token,
)
from app.services.rate_limiter import IpRateLimiter
from app.services.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

LANG_MAP = {"auto": None, "en": Language.EN, "id": Language.ID}
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
HAS_MULTIPART = importlib.util.find_spec("multipart") is not None


def _get_settings(app: FastAPI) -> Settings:
    settings = getattr(app.state, "settings", None)
    if settings is None:
        settings = load_settings()
        app.state.settings = settings
    return settings


def _get_runtime_dependencies(app: FastAPI) -> tuple[Settings, VectorStore]:
    settings = _get_settings(app)
    vector_store = getattr(app.state, "vector_store", None)

    if vector_store is None:
        vector_store = VectorStore(settings)
        vector_store.ensure_index()
        app.state.vector_store = vector_store

    return settings, vector_store


def _get_rate_limiter(app: FastAPI) -> IpRateLimiter:
    limiter = getattr(app.state, "rate_limiter", None)
    if limiter is not None:
        return limiter

    settings = _get_settings(app)

    limiter = IpRateLimiter(
        max_requests=settings.rate_limit_requests_per_hour,
        window_seconds=settings.rate_limit_window_seconds,
    )
    app.state.rate_limiter = limiter
    return limiter


def _resolve_client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        first_ip = forwarded_for.split(",", 1)[0].strip()
        if first_ip:
            return first_ip

    real_ip = request.headers.get("x-real-ip", "").strip()
    if real_ip:
        return real_ip

    client_host = request.client.host if request.client else ""
    return client_host.strip() or "unknown"


def _is_trusted_proxy_request(request: Request, settings: Settings) -> bool:
    if not settings.trusted_proxy_secret:
        return False

    provided = request.headers.get("x-gapura-proxy-secret", "")
    return hmac.compare_digest(provided, settings.trusted_proxy_secret)


def _build_evidence_payload(evidence: list[RetrievedContext]) -> list[dict[str, object]]:
    return [
        {
            "id": ctx.evidence_id,
            "source": ctx.source_filename,
            "page": ctx.page,
            "snippet": ctx.snippet,
            "score": round(ctx.score, 3),
            "rerank_score": round(ctx.rerank_score, 3)
            if ctx.rerank_score is not None
            else None,
        }
        for ctx in evidence
    ]


def _build_citations_payload(evidence: list[RetrievedContext]) -> list[dict[str, object]]:
    seen: set[tuple[str, int]] = set()
    citations: list[dict[str, object]] = []
    for ctx in evidence:
        key = (ctx.source_filename, ctx.page)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source": ctx.source_filename,
                "page": ctx.page,
                "score": round(ctx.score, 3),
            }
        )
    return citations


def _stats_value(stats: object, key: str, default: object = None) -> object:
    if isinstance(stats, dict):
        return stats.get(key, default)
    return getattr(stats, key, default)


def create_app(
    settings: Settings | None = None,
    vector_store: VectorStore | None = None,
    rate_limiter: IpRateLimiter | None = None,
) -> FastAPI:
    app = FastAPI(title="Gapura RAG API")

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # We manually log the detail because the user only sees the 422 in their terminal
        logger.error(f"Validation error for {request.url}: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(), "body": str(exc.body)},
        )
    app.state.settings = settings
    app.state.vector_store = vector_store
    app.state.rate_limiter = rate_limiter

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount(
        "/assets",
        StaticFiles(directory=str(Path(__file__).resolve().parent.parent)),
        name="assets",
    )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

    if HAS_MULTIPART:
        from fastapi import BackgroundTasks

        def _bg_ingest_pdf(
            file_bytes: bytes,
            filename: str,
            settings: Settings,
            vector_store: VectorStore,
            replace_existing: bool = False,
        ):
            try:
                logger.info("Starting background ingestion for %s", filename)
                result = ingest_pdf(
                    file_bytes,
                    filename,
                    settings,
                    vector_store,
                    replace_existing=replace_existing,
                )
                logger.info(
                    "Background ingestion completed for %s: %d pages, %d chunks",
                    filename,
                    result.total_pages,
                    result.total_chunks,
                )
            except Exception as exc:
                logger.error("Background ingestion failed for %s: %s", filename, exc)

        @app.post("/api/upload")
        async def upload_pdf(
            request: Request,
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
        ):
            try:
                settings, vector_store = _get_runtime_dependencies(request.app)

                if not file.filename or not file.filename.lower().endswith(".pdf"):
                    return JSONResponse({"error": "Only PDF files are accepted"}, status_code=400)

                file_bytes = await file.read()
                size_mb = len(file_bytes) / (1024 * 1024)
                doc_id = compute_doc_id(file_bytes)
                replace_existing = (
                    request.query_params.get("replace", "").strip().lower() == "true"
                )

                if size_mb > settings.max_pdf_size_mb:
                    return JSONResponse(
                        {"error": f"File exceeds {settings.max_pdf_size_mb}MB limit"},
                        status_code=400,
                    )

                # Check if document already exists (fast)
                if vector_store.doc_exists(doc_id) and not replace_existing:
                    return {
                        "doc_id": str(doc_id),
                        "filename": file.filename,
                        "pages": 0,
                        "chunks": 0,
                        "skipped": True,
                    }

                # Start background processing
                background_tasks.add_task(
                    _bg_ingest_pdf,
                    file_bytes,
                    file.filename,
                    settings,
                    vector_store,
                    replace_existing,
                )

                # Return immediate response. Page/chunk counts won't be known yet, 
                # so we return a placeholder or 0.
                return {
                    "doc_id": str(doc_id),
                    "filename": file.filename,
                    "pages": -1, # UI special value for "Processing..."
                    "chunks": -1,
                    "skipped": False,
                    "replace_existing": replace_existing,
                }
            except Exception as exc:
                logger.exception("Upload error for %s", file.filename)
                return JSONResponse({"error": str(exc)}, status_code=500)
    else:
        @app.post("/api/upload")
        async def upload_pdf_unavailable():
            return JSONResponse(
                {"error": "python-multipart is not installed on this deployment."},
                status_code=500,
            )

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        question = body.get("question", "").strip()
        lang = body.get("language", "auto")
        history = body.get("history", [])
        sources = body.get("sources")
        stream_mode = body.get("stream_mode", "")

        if not question:
            return JSONResponse({"error": "Question cannot be empty"}, status_code=400)

        try:
            settings = _get_settings(request.app)
        except Exception as exc:
            logger.exception("Settings bootstrap failed during chat")
            return JSONResponse({"error": str(exc)}, status_code=500)

        if _is_trusted_proxy_request(request, settings):
            pass
        elif settings.account_rate_limit_enabled:
            try:
                account_decision = consume_account_rate_limit(
                    extract_bearer_token(request.headers.get("authorization")),
                    settings,
                )
            except Exception as exc:
                logger.exception("Account rate limit check failed during chat")
                return JSONResponse({"error": str(exc)}, status_code=500)

            if not account_decision.allowed:
                content: dict[str, object] = {
                    "error": account_decision.error
                    or "Virtual Assistant rate limit exceeded.",
                }
                headers: dict[str, str] = {}
                if account_decision.retry_after_seconds:
                    content["retry_after_seconds"] = (
                        account_decision.retry_after_seconds
                    )
                    headers["Retry-After"] = str(
                        account_decision.retry_after_seconds
                    )
                if account_decision.remaining is not None:
                    content["remaining"] = account_decision.remaining
                return JSONResponse(
                    content,
                    status_code=account_decision.status_code,
                    headers=headers,
                )
        else:
            try:
                limiter = _get_rate_limiter(request.app)
            except Exception as exc:
                logger.exception("Rate limiter bootstrap failed during chat")
                return JSONResponse({"error": str(exc)}, status_code=500)

            max_requests = settings.rate_limit_requests_per_hour
            client_ip = _resolve_client_ip(request)
            limit_decision = limiter.check(client_ip)
            if not limit_decision.allowed:
                retry_after = limit_decision.retry_after_seconds
                return JSONResponse(
                    {
                        "error": (
                            "Rate limit exceeded. "
                            f"You can ask up to {max_requests} questions per hour from the same IP address."
                        ),
                        "retry_after_seconds": retry_after,
                    },
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
            )

        try:
            settings, vector_store = _get_runtime_dependencies(request.app)
        except Exception as exc:
            logger.exception("Runtime bootstrap failed during chat")
            return JSONResponse({"error": str(exc)}, status_code=500)

        lang_override = LANG_MAP.get(lang)

        def event_stream():
            try:
                yield (
                    "data: "
                    + json.dumps({"type": "status", "content": "retrieving"})
                    + "\n\n"
                )

                if stream_mode == "plain":
                    detected_language, evidence, token_stream = answer_question_plain_stream(
                        question=question,
                        settings=settings,
                        vector_store=vector_store,
                        language_override=lang_override,
                        history=history or None,
                        sources=sources,
                    )
                    answer_parts: list[str] = []
                    for token in token_stream:
                        answer_parts.append(token)
                        yield (
                            "data: "
                            + json.dumps({"type": "token", "content": token})
                            + "\n\n"
                        )

                    answer = "".join(answer_parts)
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "type": "done",
                                "answer": answer,
                                "language": detected_language.value,
                                "grounding_status": "grounded" if evidence else "unsupported",
                                "supplement_used": False,
                                "evidence": _build_evidence_payload(evidence),
                                "citations": _build_citations_payload(evidence),
                            }
                        )
                        + "\n\n"
                    )
                    return

                response: RAGResponse = answer_question(
                    question=question,
                    settings=settings,
                    vector_store=vector_store,
                    language_override=lang_override,
                    history=history or None,
                    sources=sources,
                )

                if response.answer:
                    yield (
                        "data: "
                        + json.dumps({"type": "token", "content": response.answer})
                        + "\n\n"
                    )

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "done",
                            "answer": response.answer,
                            "language": response.detected_language.value,
                            "grounding_status": response.grounding_status.value,
                            "supplement_used": response.supplement_used,
                            "evidence": _build_evidence_payload(response.evidence),
                            "citations": _build_citations_payload(
                                response.citations or response.evidence
                            ),
                        }
                    )
                    + "\n\n"
                )
            except Exception as exc:
                logger.exception("Chat error")
                yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/stats")
    async def stats(request: Request):
        try:
            settings, vector_store = _get_runtime_dependencies(request.app)
            raw = vector_store.get_stats()
            binding = vector_store.get_index_binding()
            return {
                "index": vector_store.index_name,
                "configured_index": binding["configured_index"],
                "active_index": binding["active_index"],
                "total_vectors": _stats_value(raw, "total_vector_count", 0),
                "embedding_model": settings.embedding_model,
                "embedding_dim": binding["embedding_dim"],
                "index_dimension": binding["index_dimension"],
                "pinecone_metric": binding["metric"],
                "llm_model": settings.llm_model,
            }
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    return app


app = create_app()
