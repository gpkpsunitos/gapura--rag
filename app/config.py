from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _require_env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value


def _env_or_default(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int_or_default(key: str, default: int) -> int:
    value = os.environ.get(key)
    if value is None:
        return default
    return int(value)


def _env_bool_or_default(key: str, default: bool) -> bool:
    value = os.environ.get(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    hf_token: str
    pinecone_api_key: str
    openrouter_api_key: str | None = None

    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dim: int = 1024
    chunk_size: int = 768
    chunk_overlap: int = 128

    pinecone_index: str = "gapura-rag-v2-1024d"
    pinecone_metric: str = "cosine"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    top_k: int = 3
    rerank_top_n: int = 5
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_candidate_multiplier: int = 3
    retrieval_min_score: float = 0.30
    rerank_min_score: float = 0.1
    min_supporting_evidence: int = 2
    # radius=4 means 4 chunks before and 4 chunks after for focused context
    context_window_radius: int = 4
    source_diversity_cap: int = 2
    multi_query_enabled: bool = False
    multi_query_count: int = 2

    llm_model: str = "mistralai/mistral-small-24b-instruct-2501"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    # For the CoT reasoning pass — can use a smaller/faster model
    llm_reasoning_model: str = "mistralai/mistral-small-24b-instruct-2501"
    llm_reasoning_max_tokens: int = 1024
    llm_verification_enabled: bool = False
    llm_query_decomposition_enabled: bool = False
    llm_cot_enabled: bool = False
    llm_answer_attempts: int = 1

    language_confidence_threshold: float = 0.5
    max_pdf_size_mb: int = 100
    upsert_batch_size: int = 250

    # Accuracy Enhancements
    bm25_weight: float = (
        0.3  # Weight for BM25 sparse score (0.0 = pure semantic, 1.0 = pure keyword)
    )
    hyde_enabled: bool = False
    parent_retrieval_enabled: bool = True
    rate_limit_requests_per_hour: int = 5
    rate_limit_window_seconds: int = 3600
    account_rate_limit_enabled: bool = False
    account_rate_limit_consume_url: str | None = None
    account_rate_limit_internal_secret: str | None = None
    trusted_proxy_secret: str | None = None


def _env_float_or_default(key: str, default: float) -> float:
    value = os.environ.get(key)
    if value is None:
        return default
    return float(value)


# Complexity: Time O(1) | Space O(1)
def load_settings() -> Settings:
    # Handle legacy hybrid search alpha
    hybrid_alpha = _env_float_or_default("HYBRID_SEARCH_ALPHA", 0.3)
    bm25_weight = _env_float_or_default("BM25_WEIGHT", hybrid_alpha)

    return Settings(
        hf_token=_require_env("HF_TOKEN"),
        pinecone_api_key=_require_env("PINECONE_API_KEY"),
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY"),
        embedding_model=_env_or_default(
            "EMBEDDING_MODEL",
            "intfloat/multilingual-e5-large",
        ),
        embedding_dim=_env_int_or_default("EMBEDDING_DIM", 1024),
        pinecone_index=_env_or_default("PINECONE_INDEX", "gapura-rag-v2-1024d"),
        top_k=_env_int_or_default("TOP_K", 3),
        retrieval_candidate_multiplier=_env_int_or_default(
            "RETRIEVAL_CANDIDATE_MULTIPLIER",
            3,
        ),
        source_diversity_cap=_env_int_or_default("SOURCE_DIVERSITY_CAP", 2),
        context_window_radius=_env_int_or_default("CONTEXT_WINDOW_RADIUS", 4),
        multi_query_enabled=_env_bool_or_default("MULTI_QUERY_ENABLED", False),
        multi_query_count=_env_int_or_default("MULTI_QUERY_COUNT", 2),
        bm25_weight=bm25_weight,
        hyde_enabled=_env_bool_or_default("HYDE_ENABLED", False),
        parent_retrieval_enabled=os.environ.get(
            "PARENT_RETRIEVAL_ENABLED", "true"
        ).lower()
        == "true",
        rate_limit_requests_per_hour=_env_int_or_default(
            "RATE_LIMIT_REQUESTS_PER_HOUR",
            5,
        ),
        rate_limit_window_seconds=_env_int_or_default(
            "RATE_LIMIT_WINDOW_SECONDS",
            3600,
        ),
        account_rate_limit_enabled=_env_bool_or_default(
            "ACCOUNT_RATE_LIMIT_ENABLED",
            False,
        ),
        account_rate_limit_consume_url=os.environ.get(
            "ACCOUNT_RATE_LIMIT_CONSUME_URL"
        ),
        account_rate_limit_internal_secret=os.environ.get(
            "ACCOUNT_RATE_LIMIT_INTERNAL_SECRET"
        ),
        trusted_proxy_secret=os.environ.get("TRUSTED_PROXY_SECRET"),
        llm_model=_env_or_default("LLM_MODEL", "mistralai/mistral-small-24b-instruct-2501"),
        llm_temperature=_env_float_or_default("LLM_TEMPERATURE", 0.2),
        llm_max_tokens=_env_int_or_default("LLM_MAX_TOKENS", 2048),
        llm_reasoning_model=_env_or_default(
            "LLM_REASONING_MODEL", "mistralai/mistral-small-24b-instruct-2501"
        ),
        llm_reasoning_max_tokens=_env_int_or_default("LLM_REASONING_MAX_TOKENS", 1024),
        llm_verification_enabled=_env_bool_or_default("LLM_VERIFICATION_ENABLED", False),
        llm_query_decomposition_enabled=_env_bool_or_default(
            "LLM_QUERY_DECOMPOSITION_ENABLED",
            False,
        ),
        llm_cot_enabled=_env_bool_or_default("LLM_COT_ENABLED", False),
        llm_answer_attempts=max(1, _env_int_or_default("LLM_ANSWER_ATTEMPTS", 1)),
    )
