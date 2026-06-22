from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor

from app.config import Settings
from app.models.schemas import RetrievedContext
from app.models.types import Language
from app.services.embedder import embed_query
from app.services.generator import (
    decompose_query,
    generate_hypothetical_answer,
    generate_query_variations,
)
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = {
    "apa",
    "saja",
    "dalam",
    "dan",
    "atau",
    "yang",
    "di",
    "ke",
    "of",
    "the",
    "what",
    "are",
    "list",
}

# Bilingual term map: Indonesian ↔ English for aviation/airport domain
_BILINGUAL_MAP: dict[str, set[str]] = {
    "bagasi": {"baggage", "luggage"},
    "baggage": {"bagasi"},
    "luggage": {"bagasi"},
    "penumpang": {"passenger"},
    "passenger": {"penumpang"},
    "penerbangan": {"flight"},
    "flight": {"penerbangan"},
    "bandara": {"airport"},
    "airport": {"bandara"},
    "keamanan": {"security"},
    "security": {"keamanan"},
    "keselamatan": {"safety"},
    "safety": {"keselamatan"},
    "kargo": {"cargo"},
    "cargo": {"kargo"},
    "tiket": {"ticket"},
    "ticket": {"tiket"},
    "boarding": {"naik", "boarding"},
    "check": {"cek", "check"},
    "keberangkatan": {"departure"},
    "departure": {"keberangkatan"},
    "kedatangan": {"arrival"},
    "arrival": {"kedatangan"},
    "tertunda": {"delay", "delayed"},
    "delayed": {"tertunda", "delay"},
    "irregularity": {"ketidakteraturan", "irregularitas"},
    "handling": {"penanganan"},
    "penanganan": {"handling"},
    "manajemen": {"management"},
    "management": {"manajemen"},
    "pelayanan": {"service", "services"},
    "service": {"pelayanan", "layanan"},
    "layanan": {"service", "services"},
}
_LIST_QUERY_PATTERNS = (
    re.compile(r"\bapa saja\b", re.IGNORECASE),
    re.compile(r"\bdaftar\b", re.IGNORECASE),
    re.compile(r"\blist\b", re.IGNORECASE),
    re.compile(r"\bwhat are\b", re.IGNORECASE),
    re.compile(r"\bwhich\b", re.IGNORECASE),
)
_BOILERPLATE_PATTERNS = (
    re.compile(r"\b(preface|foreword|kata pengantar)\b", re.IGNORECASE),
    re.compile(r"\b(lembar persetujuan|internal approval)\b", re.IGNORECASE),
    re.compile(r"\b(form evaluasi|vendor|outsourcing)\b", re.IGNORECASE),
    re.compile(r"\b(issue|revision|effective date)\b", re.IGNORECASE),
)
_LIST_EVIDENCE_PATTERNS = (
    re.compile(r"\bmeliputi\b", re.IGNORECASE),
    re.compile(r"\bterdiri dari\b", re.IGNORECASE),
    re.compile(r"\b1[\.\)]\s", re.IGNORECASE),
    re.compile(r"\ba\)\s", re.IGNORECASE),
)


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def preload_models(model_name: str) -> None:
    # No local reranker preload on HF Spaces free tier.
    del model_name


def _should_rerank(language: Language, settings: Settings) -> bool:
    del language
    return settings.rerank_enabled


def _context_passes_threshold(context: RetrievedContext, settings: Settings) -> bool:
    text = (context.snippet or context.text or "").strip()
    # Reject very short fragments (headers, footers, page numbers) — they are noise
    if len(text) < 50:
        return False

    dense_ok = context.score >= settings.retrieval_min_score
    if context.rerank_score is None:
        return dense_ok

    rerank_ok = context.rerank_score >= settings.rerank_min_score
    return dense_ok or rerank_ok


def _grounding_strength(
    contexts: list[RetrievedContext],
    settings: Settings,
) -> tuple[int, float, float]:
    passing = [ctx for ctx in contexts if _context_passes_threshold(ctx, settings)]
    best_dense = max((ctx.score for ctx in passing), default=0.0)
    best_rerank = max(
        (ctx.rerank_score for ctx in passing if ctx.rerank_score is not None),
        default=0.0,
    )
    return len(passing), best_dense, best_rerank


def _passes_grounding(contexts: list[RetrievedContext], settings: Settings) -> bool:
    supported_count, _, _ = _grounding_strength(contexts, settings)
    return supported_count >= settings.min_supporting_evidence


def _assign_evidence_ids(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
    return [
        ctx.model_copy(
            update={"evidence_id": f"E{idx}", "snippet": ctx.snippet or ctx.text}
        )
        for idx, ctx in enumerate(contexts, start=1)
    ]


def _source_key(context: RetrievedContext) -> str:
    if context.source_filename:
        return context.source_filename
    if context.doc_id:
        return str(context.doc_id)
    return str(context.chunk_id)


def _source_title_tokens(source_filename: str) -> set[str]:
    source_clean = re.sub(r"\.pdf$", "", source_filename.lower(), flags=re.IGNORECASE)
    base_tokens = {
        token
        for token in re.split(r"[_\s\-]+", source_clean)
        if len(token) > 2 and token not in _STOPWORDS
    }
    # Expand with bilingual equivalents
    expanded = set(base_tokens)
    for token in base_tokens:
        equivalents = _BILINGUAL_MAP.get(token)
        if equivalents:
            expanded.update(equivalents)
    return expanded


def _query_focus_tokens(query: str) -> set[str]:
    base_tokens = {
        token
        for token in _tokenize(query)
        if len(token) > 2 and token not in _STOPWORDS
    }
    # Expand with bilingual equivalents
    expanded = set(base_tokens)
    for token in base_tokens:
        equivalents = _BILINGUAL_MAP.get(token)
        if equivalents:
            expanded.update(equivalents)
    return expanded


def _is_list_query(query: str) -> bool:
    return any(pattern.search(query) for pattern in _LIST_QUERY_PATTERNS)


def _preferred_source_for_query(
    query: str,
    candidates: list[RetrievedContext],
) -> str | None:
    query_tokens = _query_focus_tokens(query)
    if not query_tokens:
        return None

    source_scores: dict[str, tuple[int, float]] = {}
    for candidate in candidates:
        source = _source_key(candidate)
        overlap = len(
            query_tokens & _source_title_tokens(candidate.source_filename or "")
        )
        if overlap <= 0:
            continue
        prev_overlap, prev_score = source_scores.get(source, (0, 0.0))
        source_scores[source] = (
            max(prev_overlap, overlap),
            max(prev_score, candidate.rerank_score or candidate.score),
        )

    if not source_scores:
        return None

    ranked_sources = sorted(
        source_scores.items(),
        key=lambda item: (item[1][0], item[1][1]),
        reverse=True,
    )
    best_source, (best_overlap, _) = ranked_sources[0]
    second_overlap = ranked_sources[1][1][0] if len(ranked_sources) > 1 else 0
    if best_overlap <= 0:
        return None
    if best_overlap >= 2 or best_overlap > second_overlap:
        return best_source
    return None


def _apply_source_diversity(
    candidates: list[RetrievedContext],
    top_k: int,
    per_source_cap: int,
    preferred_source: str | None = None,
    preferred_source_cap: int | None = None,
    min_sources: int = 2,
) -> list[RetrievedContext]:
    if per_source_cap <= 0:
        return candidates[:top_k]

    selected: list[RetrievedContext] = []
    overflow: list[RetrievedContext] = []
    counts: dict[str, int] = {}

    for candidate in candidates:
        source = _source_key(candidate)
        source_count = counts.get(source, 0)
        source_cap = per_source_cap
        if preferred_source is not None and source == preferred_source:
            source_cap = preferred_source_cap or per_source_cap
        if source_count < source_cap:
            selected.append(candidate)
            counts[source] = source_count + 1
        else:
            overflow.append(candidate)

    # Ensure minimum source diversity: add top candidate from unrepresented sources
    unique_sources = {_source_key(ctx) for ctx in selected}
    if len(unique_sources) < min_sources and overflow:
        for candidate in overflow:
            source = _source_key(candidate)
            if source not in unique_sources:
                selected.append(candidate)
                unique_sources.add(source)
                if len(unique_sources) >= min_sources:
                    break

    if len(selected) < top_k:
        selected.extend(overflow[: max(0, top_k - len(selected))])

    diversified = selected[:top_k]
    logger.info(
        "Selected %d contexts across %d sources: %s",
        len(diversified),
        len({_source_key(ctx) for ctx in diversified}),
        {
            source: sum(1 for ctx in diversified if _source_key(ctx) == source)
            for source in sorted({_source_key(ctx) for ctx in diversified})
        },
    )
    return diversified


def _dedupe_candidates(candidates: list[RetrievedContext]) -> list[RetrievedContext]:
    unique_candidates: dict[str, RetrievedContext] = {}
    for ctx in candidates:
        cid = str(ctx.chunk_id)
        if cid not in unique_candidates:
            unique_candidates[cid] = ctx
        elif ctx.score > unique_candidates[cid].score:
            unique_candidates[cid] = ctx
    return list(unique_candidates.values())


def _rank_candidates(
    query: str,
    candidates: list[RetrievedContext],
    language: Language,
    settings: Settings,
) -> list[RetrievedContext]:
    deduped = _dedupe_candidates(candidates)
    logger.info("Unified candidate pool: %d unique contexts", len(deduped))

    if not deduped:
        return []

    if _should_rerank(language, settings):
        return _rerank(query, deduped)

    return sorted(deduped, key=lambda ctx: ctx.score, reverse=True)


def _select_supported_contexts(
    query: str,
    candidates: list[RetrievedContext],
    language: Language,
    settings: Settings,
    top_k: int,
) -> list[RetrievedContext]:
    ranked = _rank_candidates(query, candidates, language, settings)
    preferred_source = _preferred_source_for_query(query, ranked)
    selected = _apply_source_diversity(
        ranked,
        top_k=top_k,
        per_source_cap=settings.source_diversity_cap,
        preferred_source=preferred_source,
        preferred_source_cap=max(top_k // 2, settings.source_diversity_cap) if preferred_source else None,
        min_sources=2,
    )
    supported_contexts = [
        ctx for ctx in selected if _context_passes_threshold(ctx, settings)
    ]
    logger.info(
        "Grounding filter kept %d contexts for query '%s' (rerank threshold=%.2f)",
        len(supported_contexts),
        query,
        settings.rerank_min_score,
    )
    return supported_contexts


def _fetch_query_candidates(
    search_queries: list[str],
    settings: Settings,
    vector_store: VectorStore,
    top_k: int,
    sources: list[str] | None,
) -> list[RetrievedContext]:
    if not search_queries:
        return []

    candidate_count_per_query = top_k * settings.retrieval_candidate_multiplier

    def fetch_batch(search_query: str) -> list[RetrievedContext]:
        try:
            emb = embed_query(search_query, settings.embedding_model)
            return vector_store.query_similar(
                embedding=emb,
                top_k=candidate_count_per_query,
                sources=sources,
            )
        except Exception as exc:
            logger.warning(
                "Retrieval failed for query variant '%s': %s",
                search_query,
                exc,
            )
            return []

    if len(search_queries) == 1:
        return fetch_batch(search_queries[0])

    with ThreadPoolExecutor(max_workers=len(search_queries)) as executor:
        results = list(executor.map(fetch_batch, search_queries))

    combined: list[RetrievedContext] = []
    for batch in results:
        combined.extend(batch)
    return combined


def _expand_supported_contexts(
    contexts: list[RetrievedContext],
    settings: Settings,
    vector_store: VectorStore,
) -> list[RetrievedContext]:
    if settings.context_window_radius <= 0:
        return contexts

    expanded = vector_store.expand_contexts(
        contexts,
        settings.context_window_radius,
    )
    seen_ranges: set[str] = set()
    unique_expanded: list[RetrievedContext] = []
    for context in expanded:
        range_key = f"{context.doc_id}_{context.chunk_index}"
        if range_key in seen_ranges:
            continue
        seen_ranges.add(range_key)
        unique_expanded.append(context)
    return unique_expanded


def retrieve(
    query: str,
    settings: Settings,
    vector_store: VectorStore,
    language: Language,
    top_k: int | None = None,
    sources: list[str] | None = None,
    expand_contexts: bool = True,
) -> list[RetrievedContext]:
    k = top_k or settings.top_k
    all_candidates: list[RetrievedContext] = []

    def evaluate_current_pool(stage_name: str) -> list[RetrievedContext]:
        supported_contexts = _select_supported_contexts(
            query=query,
            candidates=all_candidates,
            language=language,
            settings=settings,
            top_k=k,
        )
        if _passes_grounding(supported_contexts, settings):
            logger.info("Grounding succeeded after %s stage", stage_name)
            expanded = (
                _expand_supported_contexts(
                    supported_contexts,
                    settings,
                    vector_store,
                )
                if expand_contexts
                else supported_contexts
            )
            return _assign_evidence_ids(expanded)
        logger.info("Grounding remained weak after %s stage", stage_name)
        return []

    logger.info("Executing staged retrieval for query '%s'", query)

    all_candidates.extend(
        _fetch_query_candidates(
            search_queries=[query],
            settings=settings,
            vector_store=vector_store,
            top_k=k,
            sources=sources,
        )
    )
    result = evaluate_current_pool("original")
    if result:
        return result

    if settings.multi_query_enabled:
        variations = generate_query_variations(
            query,
            language,
            settings,
            settings.multi_query_count,
        )
        variation_queries = [
            variation
            for variation in dict.fromkeys(variations)
            if variation and variation.strip() != query.strip()
        ]
        if variation_queries:
            logger.info(
                "Escalating retrieval with %d query variations",
                len(variation_queries),
            )
            all_candidates.extend(
                _fetch_query_candidates(
                    search_queries=variation_queries,
                    settings=settings,
                    vector_store=vector_store,
                    top_k=k,
                    sources=sources,
                )
            )
            result = evaluate_current_pool("multi-query")
            if result:
                return result

    if settings.hyde_enabled:
        hyde_query = generate_hypothetical_answer(query, settings).strip()
        if hyde_query and hyde_query != query.strip():
            logger.info("Escalating retrieval with HyDE query")
            all_candidates.extend(
                _fetch_query_candidates(
                    search_queries=[hyde_query],
                    settings=settings,
                    vector_store=vector_store,
                    top_k=k,
                    sources=sources,
                )
            )
            result = evaluate_current_pool("HyDE")
            if result:
                return result

    # Stage 4: Query Decomposition — break complex questions into sub-queries
    if settings.llm_query_decomposition_enabled:
        sub_queries = decompose_query(query, language, settings)
        if sub_queries:
            logger.info(
                "Escalating retrieval with %d decomposed sub-queries",
                len(sub_queries),
            )
            for sub_query in sub_queries:
                all_candidates.extend(
                    _fetch_query_candidates(
                        search_queries=[sub_query],
                        settings=settings,
                        vector_store=vector_store,
                        top_k=k,
                        sources=sources,
                    )
                )
            result = evaluate_current_pool("decomposition")
            if result:
                return result

    return []


def _rerank(
    query: str,
    candidates: list[RetrievedContext],
) -> list[RetrievedContext]:
    query_tokens = _tokenize(query)
    query_lower = query.lower() or ""
    query_focus_tokens = _query_focus_tokens(query)
    list_query = _is_list_query(query)

    # Extract structural patterns from query (e.g. "2.1.2")
    structural_patterns = set(re.findall(r"\b\d+(?:\.\d+)+\b", query_lower))

    scored: list[tuple[RetrievedContext, float]] = []

    # Pre-tokenize query for phrase check
    query_words = [w for w in query_lower.split() if len(w) > 1]
    # Generate bigrams for better phrase matching
    query_bigrams: set[str] = set()
    for i in range(len(query_words) - 1):
        query_bigrams.add(f"{query_words[i]} {query_words[i + 1]}")

    for ctx in candidates:
        text_content = (ctx.snippet or ctx.text or "").lower()
        source_lower = (ctx.source_filename or "").lower()
        source_tokens = _source_title_tokens(source_lower)

        # 1. Semantic score (normalized approx 0.3-0.9)
        semantic = ctx.score

        # 2. Token overlap in text (Jaccard-like)
        text_tokens = _tokenize(text_content)
        text_overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)

        # 3. Structural Anchor Boost
        structural_anchor = 0.0
        for pat in structural_patterns:
            if pat in text_content:
                structural_anchor = 1.5  # Significant boost for matching section number
                break

        # 4. Exact Phrase Boost (bonus for sequences of 3+ words)
        phrase_boost = 0.0
        if len(query_words) >= 3:
            for i in range(len(query_words) - 2):
                phrase = " ".join(query_words[i : i + 3])
                if phrase in text_content:
                    phrase_boost = 1.0
                    break

        # 5. Bigram overlap (finer-grained than token overlap)
        bigram_overlap = 0.0
        text_bigrams: set[str] = set()
        text_words = text_content.split()
        for i in range(len(text_words) - 1):
            if len(text_words[i]) > 1 and len(text_words[i + 1]) > 1:
                text_bigrams.add(f"{text_words[i]} {text_words[i + 1]}")
        if query_bigrams:
            bigram_overlap = len(query_bigrams & text_bigrams) / max(
                len(query_bigrams), 1
            )

        # 6. Filename heuristic: Check if specific terms from the filename are in the query
        source_clean = re.sub(r"\.pdf$", "", source_lower, flags=re.IGNORECASE)
        source_parts = [p for p in re.split(r"[_\s\-]", source_clean) if len(p) > 2]

        filename_direct_boost = 0.0
        for p in source_parts:
            if p == "tahun":
                continue
            if p in query_lower:
                filename_direct_boost = 1.0
                break
            # Check bilingual equivalents
            equivalents = _BILINGUAL_MAP.get(p)
            if equivalents and any(eq in query_lower for eq in equivalents):
                filename_direct_boost = 1.0
                break

        source_overlap = len(query_focus_tokens & source_tokens)
        source_focus_boost = min(source_overlap * 0.4, 1.0)

        list_structure_boost = 0.0
        if list_query and any(
            pattern.search(text_content) for pattern in _LIST_EVIDENCE_PATTERNS
        ):
            list_structure_boost = 0.9

        # 7. Text length quality signal — penalize very short snippets (likely headers/footers)
        text_length_score = 0.0
        text_len = len(text_content.strip())
        if text_len > 200:
            text_length_score = 0.3
        elif text_len > 100:
            text_length_score = 0.1

        # 8. Number of query tokens found in text (absolute count signal)
        exact_token_count = len(query_tokens & text_tokens)
        token_count_boost = min(exact_token_count * 0.1, 0.5)

        # Unified scoring formula (Phase 5 — balanced source diversity)
        combined_score = (
            (text_overlap * 0.35)
            + (bigram_overlap * 0.25)
            + (filename_direct_boost * 0.3)
            + (source_focus_boost * 0.5)
            + (list_structure_boost * 0.8)
            + (structural_anchor * 1.5)
            + (phrase_boost * 0.8)
            + (semantic * 0.6)
            + (text_length_score * 0.3)
            + (token_count_boost * 0.3)
        )

        # 9. Table of Contents (TOC) Penalty
        # TOC entries have dot leaders (........) or "Section .... Page" patterns.
        # Use stricter detection to avoid penalizing decimal-heavy regulatory content.
        has_dot_leaders = "...." in text_content
        has_page_refs = bool(re.search(r"\.{3,}\s*\d+", text_content))
        if has_dot_leaders and has_page_refs:
            combined_score -= 1.5

        if any(pattern.search(text_content) for pattern in _BOILERPLATE_PATTERNS):
            combined_score -= 1.2

        scored.append((ctx, combined_score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [
        ctx.model_copy(update={"rerank_score": float(score)}) for ctx, score in scored
    ]
