from __future__ import annotations

import json
import logging
import re
import time
from typing import Generator

from openai import OpenAI

from app.config import Settings
from app.models.schemas import GroundedAnswerPayload, RAGResponse, RetrievedContext
from app.models.types import GroundingStatus, Language

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cached OpenRouter client singleton — avoids TCP teardown per request
# ---------------------------------------------------------------------------
_llm_client: OpenAI | None = None


def _get_llm_client(settings: Settings) -> OpenAI:
    """Return a cached OpenRouter client singleton."""
    global _llm_client
    if _llm_client is None:
        if (
            not settings.openrouter_api_key
            or "your_openrouter_api_key_here" in settings.openrouter_api_key
        ):
            raise RuntimeError(
                "OPENROUTER_API_KEY is missing or invalid in .env. "
                "Please provide a valid key from https://openrouter.ai/"
            )
        _llm_client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    return _llm_client


# ---------------------------------------------------------------------------
# System Prompts — improved for long-form, explanatory, accurate responses
# ---------------------------------------------------------------------------

_CHAT_SYSTEM_PROMPT_EN = """You are "I'm in Charge", a bilingual virtual assistant for Gapura Airport Services.

Rules:
- Reply in the same language as the user.
- For greetings, short chit-chat, and "what can you do?" questions, answer naturally and briefly.
- Do not invent document facts when no document evidence is provided.
- Maintain professional and accurate tone. For procedural or regulatory questions, provide high-detail, long-form responses mirroring the source structure.
- When explaining concepts, break them down step by step. Define technical terms and provide context."""

_CHAT_SYSTEM_PROMPT_ID = """Anda adalah "I'm in Charge", asisten virtual bilingual untuk Gapura Airport Services.

Aturan:
- Jawab dalam bahasa yang sama dengan pengguna.
- Untuk sapaan, obrolan singkat, dan pertanyaan seperti "kamu bisa apa?", jawab secara natural dan singkat.
- Jangan mengarang fakta dokumen ketika tidak ada bukti dokumen.
- Gunakan nada yang profesional dan akurat. Untuk pertanyaan prosedural atau regulasi, berikan jawaban yang mendalam dan panjang sesuai struktur sumber.
- Saat menjelaskan konsep, uraikan langkah demi langkah. Definisikan istilah teknis dan berikan konteks."""

# ---------------------------------------------------------------------------
# Chain-of-Thought (CoT) Reasoning Prompt — generates reasoning before answer
# ---------------------------------------------------------------------------

_COT_SYSTEM_PROMPT_EN = """You are an expert document analyst for Gapura Airport Services. Your task is to reason step-by-step through the provided evidence before answering.

Think through this systematically:
1. Identify what the user is asking for.
2. Find ALL relevant passages in the evidence that relate to the question.
3. For each relevant passage, note what information it provides — quote the exact text.
4. Cross-reference passages to build a complete picture.
5. Identify any gaps where the evidence does NOT fully address the question.

After your reasoning, produce the final answer.

ANTI-HALLUCINATION RULES (STRICT — violations are critical errors):
- Every factual claim MUST be directly traceable to specific evidence text. If you cannot find the exact words in the evidence, do NOT include the claim.
- Do NOT infer, extrapolate, or combine partial information to create new facts that are not explicitly stated.
- When listing steps or items, only include those EXPLICITLY stated in the evidence — never add implied, assumed, or "common sense" steps.
- If the evidence does not contain information to answer part of the question, explicitly state: "The evidence does not cover [specific topic]."
- If you are unsure whether a detail is in the evidence, OMIT it entirely.
- Do NOT paraphrase in a way that changes the meaning or adds specificity not present in the original text.
- When numbers, quantities, or time limits are mentioned, use the EXACT values from the evidence — never approximate.

Rules:
- Be thorough: extract and present ALL relevant details from the evidence.
- For procedures: list EVERY step mentioned in the evidence, in order.
- For lists/requirements: include EVERY item, do not summarize or skip.
- For explanations: break down complex concepts into clear, digestible parts.
- Use the same language as the user.
- Every factual claim must cite the evidence using [E1], [E2] etc. notation.
- Do NOT add information not present in the evidence.

Return ONLY valid JSON:
{
  "reasoning": "your step-by-step analysis of the evidence",
  "grounding_status": "grounded" | "partial" | "unsupported",
  "answer": "comprehensive answer with [E1] citations",
  "cited_evidence_ids": ["E1", "E2"],
  "supplement": null
}

Do NOT wrap the JSON in markdown fences."""

_COT_SYSTEM_PROMPT_ID = """Anda adalah analis dokumen ahli untuk Gapura Airport Services. Tugas Anda adalah menalar secara bertahap melalui bukti yang diberikan sebelum menjawab.

Pikirkan ini secara sistematis:
1. Identifikasi apa yang ditanyakan pengguna.
2. Temukan SEMUA bagian relevan dalam bukti yang terkait dengan pertanyaan.
3. Untuk setiap bagian relevan, catat informasi apa yang diberikan — kutip teks aslinya.
4. Referensi-silang bagian-bagian untuk membangun gambaran lengkap.
5. Identifikasi celah di mana bukti TIDAK sepenuhnya menjawab pertanyaan.

Setelah penalaran Anda, hasilkan jawaban akhir.

ATURAN ANTI-HALUSINASI (KETAT — pelanggaran adalah kesalahan kritis):
- Setiap klaim faktual HARUS dapat dilacak langsung ke teks bukti spesifik. Jika Anda tidak dapat menemukan kata-kata yang tepat dalam bukti, JANGAN sertakan klaim tersebut.
- JANGAN menyimpulkan, mengekstrapolasi, atau menggabungkan informasi parsial untuk membuat fakta baru yang tidak dinyatakan secara eksplisit.
- Saat mendaftar langkah atau item, hanya sertakan yang DINYATAKAN SECARA EKSPLISIT dalam bukti — jangan menambahkan langkah yang tersirat, diasumsikan, atau "akal sehat".
- Jika bukti tidak mengandung informasi untuk menjawab sebagian pertanyaan, nyatakan secara eksplisit: "Bukti tidak mencakup [topik spesifik]."
- Jika Anda tidak yakin apakah suatu detail ada dalam bukti, HAPUS sepenuhnya.
- JANGAN parafrase dengan cara yang mengubah makna atau menambahkan spesifisitas yang tidak ada dalam teks asli.
- Ketika angka, kuantitas, atau batasan waktu disebutkan, gunakan nilai yang TEPAT dari bukti — jangan memperkirakan.

Aturan:
- Berikan secara menyeluruh: ekstrak dan sajikan SEMUA detail relevan dari bukti.
- Untuk prosedur: daftar SETIAP langkah yang disebutkan dalam bukti, secara berurutan.
- Untuk daftar/persyaratan: sertakan SETIAP item, jangan merangkum atau melewatkan.
- Untuk penjelasan: uraikan konsep kompleks menjadi bagian-bagian yang jelas dan mudah dipahami.
- Gunakan bahasa yang sama dengan pengguna.
- Setiap pernyataan faktual harus menyertakan bukti menggunakan notasi [E1], [E2] dll.
- JANGAN menambahkan informasi yang tidak ada dalam bukti.

Kembalikan HANYA JSON valid:
{
  "reasoning": "analisis bertahap Anda terhadap bukti",
  "grounding_status": "grounded" | "partial" | "unsupported",
  "answer": "jawaban komprehensif dengan sitasi [E1]",
  "cited_evidence_ids": ["E1", "E2"],
  "supplement": null
}

JANGAN bungkus JSON dengan markdown fences."""

# ---------------------------------------------------------------------------
# Grounding system prompt — kept as fallback for simpler queries
# ---------------------------------------------------------------------------

_GROUNDING_SYSTEM_PROMPT_EN = """You are "I'm in Charge", a grounded document assistant for Gapura Airport Services.

Return ONLY valid JSON with this exact shape:
{
  "grounding_status": "grounded" | "partial" | "unsupported",
  "answer": "string",
  "cited_evidence_ids": ["E1", "E2"],
  "supplement": "string or null"
}

ANTI-HALLUCINATION RULES (STRICT — violations are critical errors):
- Every factual claim MUST be directly traceable to specific evidence text. If you cannot find the exact words in the evidence, do NOT include the claim.
- Do NOT infer, extrapolate, or combine partial information to create new facts.
- When listing steps or items, only include those EXPLICITLY stated in the evidence — never add implied or assumed steps.
- If the evidence does not contain information for part of the question, explicitly state: "The evidence does not cover [topic]." and set grounding_status to "partial".
- If you are unsure whether a detail is in the evidence, OMIT it entirely.
- When numbers, quantities, or time limits are mentioned, use the EXACT values from the evidence.

Rules:
- Reply in the same language as the user.
- Use only the provided evidence for the main answer.
- Every factual statement in "answer" must include inline citations like [E1].
- Never add facts that are not supported by the provided evidence.
- If the evidence only partially answers the question, set "grounding_status" to "partial", answer only the supported portion, and set "supplement" to null.
- For multi-step procedures, requirements, or lists (e.g., sections a, b, c... or 1, 2, 3...), you MUST extract and provide the full exhaustive list. Do not summarise or skip items.
- Mirror the formatting and structure (lists, bullets) of the source document.
- Provide COMPREHENSIVE, DETAILED answers. Explain concepts thoroughly. Do not be terse.
- Do not wrap the JSON in markdown fences."""

_GROUNDING_SYSTEM_PROMPT_ID = """Anda adalah "I'm in Charge", asisten dokumen yang harus selalu terikat pada bukti dokumen untuk Gapura Airport Services.

Kembalikan HANYA JSON valid dengan bentuk persis seperti ini:
{
  "grounding_status": "grounded" | "partial" | "unsupported",
  "answer": "string",
  "cited_evidence_ids": ["E1", "E2"],
  "supplement": "string atau null"
}

ATURAN ANTI-HALUSINASI (KETAT — pelanggaran adalah kesalahan kritis):
- Setiap klaim faktual HARUS dapat dilacak langsung ke teks bukti spesifik. Jika Anda tidak dapat menemukan kata-kata yang tepat dalam bukti, JANGAN sertakan klaim tersebut.
- JANGAN menyimpulkan, mengekstrapolasi, atau menggabungkan informasi parsial untuk membuat fakta baru.
- Saat mendaftar langkah atau item, hanya sertakan yang DINYATAKAN SECARA EKSPLISIT dalam bukti — jangan menambahkan langkah yang tersirat atau diasumsikan.
- Jika bukti tidak mengandung informasi untuk sebagian pertanyaan, nyatakan secara eksplisit: "Bukti tidak mencakup [topik]." dan setel grounding_status ke "partial".
- Jika Anda tidak yakin apakah suatu detail ada dalam bukti, HAPUS sepenuhnya.
- Ketika angka, kuantitas, atau batasan waktu disebutkan, gunakan nilai yang TEPAT dari bukti.

Aturan:
- Jawab dalam bahasa yang sama dengan pengguna.
- Gunakan hanya bukti yang diberikan untuk jawaban utama.
- Setiap pernyataan faktual di "answer" harus punya sitasi inline seperti [E1].
- Jangan pernah menambahkan fakta yang tidak didukung oleh bukti yang diberikan.
- Jika bukti hanya menjawab sebagian pertanyaan, setel "grounding_status" menjadi "partial", jawab hanya bagian yang didukung bukti, dan setel "supplement" ke null.
- Untuk prosedur multi-step, persyaratan, atau daftar (contoh: bagian a, b, c... atau 1, 2, 3...), Anda WAJIB mengekstrak dan memberikan daftar lengkap secara lengkap. Jangan merangkum atau melewatkan item.
- Ikuti format dan struktur (daftar, bulet) dari dokumen sumber.
- Berikan jawaban yang KOMPREHENSIF dan DETAIL. Jelaskan konsep secara menyeluruh. Jangan ringkas.
- Jangan bungkus JSON dengan markdown fences."""

# ---------------------------------------------------------------------------
# Self-Verification Prompt — validates answer quality before returning
# ---------------------------------------------------------------------------

_VERIFICATION_PROMPT_EN = """You are a strict fact-checker for a document-grounded AI assistant. Your job is to extract EVERY factual claim from the proposed answer and verify each one against the evidence. You must be skeptical — flag anything not directly supported.

Evidence:
{evidence}

Question: {question}

Proposed answer:
{answer}

STEP 1: Extract every factual claim from the answer. A factual claim is any specific statement about procedures, requirements, numbers, names, rules, steps, or document contents.

STEP 2: For each claim, search the evidence for the EXACT or near-exact supporting text.

STEP 3: Classify each claim:
- "supported": The exact or near-exact text exists in the evidence. Quote the supporting text and cite the evidence ID.
- "unsupported": The claim is NOT found in the evidence, or the evidence only partially matches and the claim adds new specifics. This is a HALLUCINATION.
- "vague": The claim is too general to verify (e.g., "the document discusses various topics").

Return ONLY valid JSON:
{{
  "claims": [
    {{"claim": "the factual claim text", "verdict": "supported|unsupported|vague", "evidence_text": "exact quote from evidence or null", "evidence_id": "E1 or null"}}
  ],
  "has_hallucination": true/false,
  "issues": ["list of unsupported claims, or empty list"],
  "improved_answer": "null if all claims supported, or the answer with unsupported claims removed (keep citations)"
}}

Do NOT wrap the JSON in markdown fences."""

_VERIFICATION_PROMPT_ID = """Anda adalah pemeriksa fakta ketat untuk asisten AI yang berbasis dokumen. Tugas Anda adalah mengekstrak SETIAP klaim faktual dari jawaban yang diajukan dan memverifikasi masing-masing terhadap bukti. Anda harus skeptis — tandai apa pun yang tidak didukung secara langsung.

Bukti:
{evidence}

Pertanyaan: {question}

Jawaban yang diajukan:
{answer}

LANGKAH 1: Ekstrak setiap klaim faktual dari jawaban. Klaim faktual adalah pernyataan spesifik tentang prosedur, persyaratan, angka, nama, aturan, langkah, atau isi dokumen.

LANGKAH 2: Untuk setiap klaim, cari dalam bukti teks pendukung yang TEPAT atau hampir tepat.

LANGKAH 3: Klasifikasikan setiap klaim:
- "supported": Teks yang tepat atau hampir tepat ada dalam bukti. Kutip teks pendukung dan sebutkan ID bukti.
- "unsupported": Klaim TIDAK ditemukan dalam bukti, atau bukti hanya cocok sebagian dan klaim menambahkan spesifikasi baru. Ini adalah HALUSINASI.
- "vague": Klaim terlalu umum untuk diverifikasi (misalnya, "dokumen membahas berbagai topik").

Kembalikan HANYA JSON valid:
{{
  "claims": [
    {{"claim": "teks klaim faktual", "verdict": "supported|unsupported|vague", "evidence_text": "kutipan tepat dari bukti atau null", "evidence_id": "E1 atau null"}}
  ],
  "has_hallucination": true/false,
  "issues": ["daftar klaim tidak didukung, atau daftar kosong"],
  "improved_answer": "null jika semua klaim didukung, atau jawaban dengan klaim tidak didukung dihapus (pertahankan sitasi)"
}}

JANGAN bungkus JSON dengan markdown fences."""

# ---------------------------------------------------------------------------
# Query Decomposition Prompt — breaks complex questions into sub-queries
# ---------------------------------------------------------------------------

_DECOMPOSITION_PROMPT_EN = """Break down this complex question into 2-4 simpler sub-questions that can each be answered independently from documents. Each sub-question should target a specific aspect of the original question.

If the question is simple and does not need decomposition, return an empty list.

Question: {query}

Return ONLY valid JSON:
{{"sub_queries": ["sub-question 1", "sub-question 2"]}}

or if no decomposition needed:
{{"sub_queries": []}}

Do NOT wrap the JSON in markdown fences."""

_DECOMPOSITION_PROMPT_ID = """Uraikan pertanyaan kompleks ini menjadi 2-4 sub-pertanyaan yang lebih sederhana yang masing-masing dapat dijawab secara independen dari dokumen. Setiap sub-pertanyaan harus menargetkan aspek tertentu dari pertanyaan asli.

Jika pertanyaan sederhana dan tidak perlu diuraikan, kembalikan daftar kosong.

Pertanyaan: {query}

Kembalikan HANYA JSON valid:
{{"sub_queries": ["sub-pertanyaan 1", "sub-pertanyaan 2"]}}

atau jika tidak perlu penguraian:
{{"sub_queries": []}}

JANGAN bungkus JSON dengan markdown fences."""

_REFORMULATE_PROMPT = """Given the conversation history and the latest user message, reformulate the user's message into a clear, standalone search query that would retrieve relevant document passages.

Rules:
- Resolve pronouns and references using conversation history.
- Keep the original language (English or Bahasa Indonesia).
- If the message is already a clear standalone question, return it as-is.
- If it's a greeting or small talk, return: CHITCHAT.
- Output ONLY the reformulated query.

Conversation history:
{history}

Latest message: {message}

Reformulated query:"""

_QUERY_VARIATIONS_PROMPT = """Generate {count} different search queries in {language} that are semantically similar to the following query but use different phrasing or keywords.
These queries will be used to retrieve relevant documents from a vector database.

Original query: {query}

Output only the queries, one per line. Do not include numbering or extra text."""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_CITATION_RE = re.compile(r"\[(E\d+)\]")
_GROUNDING_STATUS_ALIASES = {
    "supported": GroundingStatus.GROUNDED.value,
}

_NO_CONTEXT_EN = (
    "I couldn't find enough support in the uploaded documents to answer that. "
    "Try asking about the uploaded documents or upload a document that covers this topic."
)
_NO_CONTEXT_ID = (
    "Saya tidak menemukan dukungan yang cukup di dokumen yang diunggah untuk menjawab pertanyaan itu. "
    "Coba tanyakan hal yang tercakup di dokumen atau unggah dokumen yang relevan."
)
_PARTIAL_WARNING_EN = (
    "Warning: Some parts of this question are not covered by the uploaded documents."
)
_PARTIAL_WARNING_ID = (
    "Peringatan: Sebagian pertanyaan ini tidak tercakup dalam dokumen yang diunggah."
)

_JSON_OPENERS = {"{", "["}
_LIST_QUERY_PATTERNS = (
    re.compile(r"\bapa saja\b", re.IGNORECASE),
    re.compile(r"\bdaftar\b", re.IGNORECASE),
    re.compile(r"\bwhich\b", re.IGNORECASE),
    re.compile(r"\bwhat are\b", re.IGNORECASE),
    re.compile(r"\blist\b", re.IGNORECASE),
)
_LIST_STOPWORDS = {
    "apa",
    "saja",
    "dalam",
    "dan",
    "atau",
    "yang",
    "di",
    "ke",
    "untuk",
    "the",
    "what",
    "are",
    "list",
}
_PROCEDURE_QUERY_PATTERNS = (
    re.compile(r"\bsop\b", re.IGNORECASE),
    re.compile(r"\bprosedur\b", re.IGNORECASE),
    re.compile(r"\bprocedure\b", re.IGNORECASE),
)
_SECTION_CAPTURE_RE = re.compile(
    r"(?:^|[\s;:])(?:\d{1,2}|[a-z])[\.\)]\s*([A-Z][A-Za-z/&\-\s]{3,80})"
)
_SECTION_SPLIT_RE = re.compile(r"[\n\r]+|(?<=[\.\)])\s+(?=(?:\d{1,2}|[a-z])[\.\)])")
_SECTION_SKIP_PATTERNS = (
    re.compile(r"\b(preface|foreword|kata pengantar)\b", re.IGNORECASE),
    re.compile(
        r"\b(approval|persetujuan|effective date|issue|revision)\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(daftar isi|table of contents|referensi|references)\b", re.IGNORECASE
    ),
    re.compile(r"\b(tanggung jawab|responsibilities)\b", re.IGNORECASE),
)
_SECTION_COMMAND_PATTERNS = (
    re.compile(
        r"^(always|prepare|providing|memberikan|persiapkan|petugas)\b", re.IGNORECASE
    ),
    re.compile(r"^(the\s+[a-z]+\s+officer)\b", re.IGNORECASE),
)

# Patterns to detect complex questions that benefit from CoT
_COMPLEX_QUERY_INDICATORS = (
    re.compile(
        r"\b(explain|jelaskan|describe|uraikan|how does|bagaimana)\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(why|mengapa|kenapa|what is|apa itu|apa yang dimaksud)\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(compare|bandingkan|difference|perbedaan|versus|vs)\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(steps|langkah|proses|process|procedure|prosedur|requirement|persyaratan)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(detail|rinci|selengkapnya|comprehensive|lengkap)\b", re.IGNORECASE),
)


def _is_complex_query(question: str) -> bool:
    """Detect questions that benefit from chain-of-thought reasoning."""
    normalized = question.strip().lower()
    return any(pattern.search(normalized) for pattern in _COMPLEX_QUERY_INDICATORS)


def _chat_system_prompt(language: Language) -> str:
    return _CHAT_SYSTEM_PROMPT_ID if language == Language.ID else _CHAT_SYSTEM_PROMPT_EN


def _grounding_system_prompt(language: Language) -> str:
    return (
        _GROUNDING_SYSTEM_PROMPT_ID
        if language == Language.ID
        else _GROUNDING_SYSTEM_PROMPT_EN
    )


def _cot_system_prompt(language: Language) -> str:
    return _COT_SYSTEM_PROMPT_ID if language == Language.ID else _COT_SYSTEM_PROMPT_EN


def _verification_prompt(language: Language) -> str:
    return (
        _VERIFICATION_PROMPT_ID if language == Language.ID else _VERIFICATION_PROMPT_EN
    )


def _decomposition_prompt(language: Language) -> str:
    return (
        _DECOMPOSITION_PROMPT_ID
        if language == Language.ID
        else _DECOMPOSITION_PROMPT_EN
    )


def _build_history_messages(
    history: list[dict[str, str]] | None,
    max_messages: int = 6,
) -> list[dict[str, str]]:
    if not history:
        return []
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history[-max_messages:]
        if msg.get("role") in {"user", "assistant"} and msg.get("content")
    ]


def _build_evidence_block(contexts: list[RetrievedContext]) -> str:
    return "\n\n".join(
        (
            f"{ctx.evidence_id}\n"
            f"Source: {ctx.source_filename}, Page {ctx.page}\n"
            f"{ctx.text.strip()}"
        )
        for ctx in contexts
    )


def _unsupported_message(language: Language) -> str:
    return _NO_CONTEXT_ID if language == Language.ID else _NO_CONTEXT_EN


def _partial_warning(language: Language) -> str:
    return _PARTIAL_WARNING_ID if language == Language.ID else _PARTIAL_WARNING_EN


def _response_content(response: object) -> str:
    try:
        if hasattr(response, "choices"):
            return str(response.choices[0].message.content or "").strip()
        return ""
    except Exception:
        return ""


def _extract_json_payload(raw_content: str) -> dict:
    """Extract the last JSON block from raw model output."""
    matches = list(_JSON_BLOCK_RE.finditer(raw_content))
    if not matches:
        raise ValueError("Model response did not contain JSON")
    last_block = matches[-1].group(0)
    try:
        parsed = json.loads(last_block)
        if isinstance(parsed, dict):
            status = parsed.get("grounding_status")
            if isinstance(status, str):
                normalized = _GROUNDING_STATUS_ALIASES.get(status.strip().lower())
                if normalized:
                    parsed["grounding_status"] = normalized
        return parsed
    except Exception as e:
        raise ValueError(f"Failed to parse last JSON block: {e}")


def _extract_grounded_payload(raw_content: str) -> GroundedAnswerPayload:
    """Extract a GroundedAnswerPayload from raw model output."""
    parsed = _extract_json_payload(raw_content)
    # CoT responses have a "reasoning" field — we don't need it in the payload
    parsed.pop("reasoning", None)
    return GroundedAnswerPayload.model_validate(parsed)


def _extract_answer_citation_ids(answer: str) -> list[str]:
    seen: set[str] = set()
    ordered_ids: list[str] = []
    for citation_id in _CITATION_RE.findall(answer):
        if citation_id in seen:
            continue
        seen.add(citation_id)
        ordered_ids.append(citation_id)
    return ordered_ids


def _append_citation_suffix(answer: str, citation_ids: list[str]) -> str:
    clean_answer = answer.rstrip()
    if not clean_answer or not citation_ids:
        return clean_answer
    suffix = " " + " ".join(f"[{citation_id}]" for citation_id in citation_ids)
    return f"{clean_answer}{suffix}"


def _normalize_payload(
    payload: GroundedAnswerPayload,
    contexts: list[RetrievedContext],
    language: Language,
) -> GroundedAnswerPayload:
    answer = payload.answer.strip()
    allowed_ids = {ctx.evidence_id for ctx in contexts}
    answer_ids = _extract_answer_citation_ids(answer)
    payload_ids = [
        citation_id
        for citation_id in payload.cited_evidence_ids
        if citation_id in allowed_ids
    ]
    invalid_ids = [aid for aid in answer_ids if aid not in allowed_ids]

    if payload.grounding_status == GroundingStatus.UNSUPPORTED:
        return payload.model_copy(
            update={
                "answer": _unsupported_message(language),
                "cited_evidence_ids": [],
                "supplement": None,
            }
        )

    if not answer:
        raise ValueError("Grounded answer was empty")

    if invalid_ids:
        raise ValueError(
            f"Grounded answer cited unknown evidence ids: {', '.join(invalid_ids)}"
        )

    if payload.grounding_status in {GroundingStatus.GROUNDED, GroundingStatus.PARTIAL}:
        if not answer_ids:
            if payload_ids:
                answer = _append_citation_suffix(answer, payload_ids)
                answer_ids = payload_ids
            else:
                raise ValueError("Grounded answer is missing valid inline citations")

    return payload.model_copy(
        update={
            "answer": answer,
            "cited_evidence_ids": answer_ids,
            "supplement": None,
        }
    )


def _validate_citations_in_answer(
    answer: str,
    contexts: list[RetrievedContext],
    language: Language,
) -> str:
    """Post-process answer: validate citations exist and flag uncited paragraphs."""
    allowed_ids = {ctx.evidence_id for ctx in contexts}

    # Remove citations that reference non-existent evidence IDs
    def _replace_invalid_citation(match: re.Match) -> str:
        cid = match.group(1)
        if cid in allowed_ids:
            return match.group(0)
        logger.warning("Removing invalid citation [%s] — not in evidence", cid)
        return ""

    cleaned = _CITATION_RE.sub(_replace_invalid_citation, answer)

    # Flag paragraphs with zero citations as potentially unsupported
    paragraphs = cleaned.split("\n\n")
    flagged_paragraphs: list[str] = []
    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            flagged_paragraphs.append(para)
            continue
        # Skip short paragraphs (headers, separators)
        if len(para_stripped) < 30:
            flagged_paragraphs.append(para)
            continue
        has_citation = bool(_CITATION_RE.search(para_stripped))
        if not has_citation:
            # Add a subtle marker that this paragraph lacks citations
            if language == Language.ID:
                flagged_paragraphs.append(para + " *(tidak ada sitasi)*")
            else:
                flagged_paragraphs.append(para + " *(no citation)*")
            logger.info("Flagged uncited paragraph: %s...", para_stripped[:80])
        else:
            flagged_paragraphs.append(para)

    return "\n\n".join(flagged_paragraphs)


def _compose_answer_text(
    payload: GroundedAnswerPayload,
    language: Language,
) -> tuple[str, bool]:
    if payload.grounding_status != GroundingStatus.PARTIAL:
        return payload.answer.strip(), False
    return f"{payload.answer.strip()}\n\n{_partial_warning(language)}", False


def _filter_cited_contexts(
    contexts: list[RetrievedContext],
    cited_evidence_ids: list[str],
) -> list[RetrievedContext]:
    cited_lookup = set(cited_evidence_ids)
    return [ctx for ctx in contexts if ctx.evidence_id in cited_lookup]


def _is_listing_question(question: str) -> bool:
    normalized = question.strip().lower()
    return any(pattern.search(normalized) for pattern in _LIST_QUERY_PATTERNS)


def _is_procedure_listing_question(question: str) -> bool:
    normalized = question.strip().lower()
    return _is_listing_question(normalized) and any(
        pattern.search(normalized) for pattern in _PROCEDURE_QUERY_PATTERNS
    )


def _source_title(source_filename: str) -> str:
    clean = re.sub(r"\.pdf$", "", source_filename, flags=re.IGNORECASE)
    clean = re.sub(r"[_\-]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean or source_filename


def _focus_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.lower())
        if len(token) > 2 and token not in _LIST_STOPWORDS
    }


def _clean_section_title(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", title).strip(" -.:;")
    if len(cleaned) < 4:
        return ""
    if cleaned.isupper():
        cleaned = cleaned.title()
    return cleaned


def _section_title_score(title: str) -> float:
    words = [word for word in title.split() if word]
    if len(words) < 2 or len(words) > 8:
        return -1.0
    if any(pattern.search(title) for pattern in _SECTION_COMMAND_PATTERNS):
        return -1.0

    title_case_words = sum(1 for word in words if word[:1].isupper())
    title_case_ratio = title_case_words / max(len(words), 1)
    length_bonus = 1.0 if 2 <= len(words) <= 5 else 0.4
    return title_case_ratio + length_bonus


def _extract_section_items(
    contexts: list[RetrievedContext],
) -> list[tuple[str, RetrievedContext]]:
    extracted: list[tuple[str, RetrievedContext]] = []
    seen_titles: set[str] = set()

    for context in contexts:
        text = (context.text or context.snippet or "").strip()
        if not text:
            continue

        segments = _SECTION_SPLIT_RE.split(text)
        best_for_context: tuple[str, RetrievedContext, float] | None = None
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            matches = list(_SECTION_CAPTURE_RE.finditer(segment))
            if not matches:
                continue
            title = _clean_section_title(matches[-1].group(1))
            if not title:
                continue
            normalized = title.lower()
            if normalized in seen_titles:
                continue
            if any(pattern.search(title) for pattern in _SECTION_SKIP_PATTERNS):
                continue
            score = _section_title_score(title)
            if score < 0:
                continue
            if best_for_context is None or score > best_for_context[2]:
                best_for_context = (title, context, score)

        if best_for_context is not None:
            title, best_context, _ = best_for_context
            seen_titles.add(title.lower())
            extracted.append((title, best_context))

    return extracted


def _synthesize_procedure_listing_answer(
    question: str,
    contexts: list[RetrievedContext],
    language: Language,
    settings: Settings,
) -> RAGResponse | None:
    if not contexts or not _is_procedure_listing_question(question):
        return None

    query_tokens = _focus_tokens(question)
    items = _extract_section_items(contexts)

    rows: list[tuple[str, RetrievedContext]] = []
    seen_titles: set[str] = set()

    for context in contexts:
        source_title = _source_title(context.source_filename)
        source_tokens = _focus_tokens(source_title)
        if rows and not (query_tokens & source_tokens):
            continue
        normalized = source_title.lower()
        if normalized in seen_titles:
            continue
        seen_titles.add(normalized)
        rows.append((source_title, context))

    for title, context in items:
        title_tokens = _focus_tokens(title)
        if query_tokens and not (query_tokens & title_tokens):
            continue
        normalized = title.lower()
        if normalized in seen_titles:
            continue
        seen_titles.add(normalized)
        rows.append((title, context))

    if len(rows) < 2:
        return None

    if language == Language.ID:
        intro = "Dari bukti yang ditemukan, bagian SOP/prosedur yang teridentifikasi meliputi:"
    else:
        intro = "Based on the retrieved evidence, the identified SOP/procedure sections are:"

    answer_lines = [
        f"{idx}. {title} [{context.evidence_id}]"
        for idx, (title, context) in enumerate(rows[:5], start=1)
    ]
    answer = f"{intro}\n" + "\n".join(answer_lines)
    answer = f"{answer}\n\n{_partial_warning(language)}"

    cited_ids = {context.evidence_id for _, context in rows[:5]}
    cited_contexts = [ctx for ctx in contexts if ctx.evidence_id in cited_ids]

    return RAGResponse(
        answer=answer,
        detected_language=language,
        citations=cited_contexts,
        evidence=contexts,
        grounding_status=GroundingStatus.PARTIAL,
        supplement_used=False,
        model_used=settings.llm_model,
    )


def _synthesize_listing_answer(
    question: str,
    contexts: list[RetrievedContext],
    language: Language,
    settings: Settings,
) -> RAGResponse | None:
    if not contexts or not _is_listing_question(question):
        return None

    source_rows: list[RetrievedContext] = []
    seen_sources: set[str] = set()
    wants_sop = "sop" in question.lower()

    for context in contexts:
        source_name = context.source_filename or ""
        title = _source_title(source_name)
        if wants_sop and "sop" not in title.lower():
            continue
        if source_name in seen_sources:
            continue
        seen_sources.add(source_name)
        source_rows.append(context)

    if not source_rows:
        return None

    if language == Language.ID:
        intro = "Dokumen yang teridentifikasi dari bukti yang tersedia meliputi:"
    else:
        intro = "The documents identified from the available evidence are:"

    bullet_lines = [
        f"- {_source_title(context.source_filename)} [{context.evidence_id}]"
        for context in source_rows
    ]
    answer = f"{intro}\n" + "\n".join(bullet_lines)
    answer = f"{answer}\n\n{_partial_warning(language)}"

    return RAGResponse(
        answer=answer,
        detected_language=language,
        citations=source_rows,
        evidence=source_rows,
        grounding_status=GroundingStatus.PARTIAL,
        supplement_used=False,
        model_used=settings.llm_model,
    )


def _build_grounded_messages(
    question: str,
    contexts: list[RetrievedContext],
    language: Language,
    history: list[dict[str, str]] | None = None,
    validation_feedback: str | None = None,
    use_cot: bool = False,
) -> list[dict[str, str]]:
    system_prompt = (
        _cot_system_prompt(language) if use_cot else _grounding_system_prompt(language)
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(_build_history_messages(history))

    evidence_block = _build_evidence_block(contexts)
    user_content = (
        "CRITICAL: Every factual claim in your answer MUST be directly traceable to the evidence below. "
        "If you cannot find support for a claim in this evidence, do NOT include it. "
        "State explicitly what the evidence does NOT cover.\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"Latest user question: {question}\n\n"
        "Use the evidence above to answer comprehensively. "
        "Provide a detailed, well-structured response with [E1] style citations for every factual claim. "
        "Return the response in the specified JSON format. "
        "Ensure the 'answer' is comprehensive, detailed, and not empty. "
        "Keep 'supplement' as null."
    )
    if validation_feedback:
        user_content += f"\n\nRepair instruction: {validation_feedback}"
    messages.append({"role": "user", "content": user_content})
    return messages


def _build_chitchat_messages(
    question: str,
    language: Language,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _chat_system_prompt(language)}
    ]
    messages.extend(_build_history_messages(history))
    messages.append({"role": "user", "content": question})
    return messages


# ---------------------------------------------------------------------------
# LLM completion helpers
# ---------------------------------------------------------------------------


def _chat_completion_stream(
    settings: Settings,
    messages: list[dict[str, str]],
    model_override: str | None = None,
    max_tokens_override: int | None = None,
) -> Generator[str, None, None]:
    client = _get_llm_client(settings)
    try:
        stream = client.chat.completions.create(
            model=model_override or settings.llm_model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=max_tokens_override or settings.llm_max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as exc:
        logger.error("OpenRouter Inference streaming failed: %s", exc)
        raise


def _chat_completion(
    settings: Settings,
    messages: list[dict[str, str]],
    max_retries: int = 2,
    model_override: str | None = None,
    max_tokens_override: int | None = None,
) -> str:
    client = _get_llm_client(settings)
    model = model_override or settings.llm_model
    max_tokens = max_tokens_override or settings.llm_max_tokens
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=settings.llm_temperature,
                max_tokens=max_tokens,
            )
            return _response_content(response)
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                logger.error(
                    "OpenRouter Inference failed after %d attempts: %s", max_retries, exc
                )
                raise
    return ""


# ---------------------------------------------------------------------------
# Self-Verification — validates answer quality before returning
# ---------------------------------------------------------------------------


def _verify_answer(
    question: str,
    answer: str,
    contexts: list[RetrievedContext],
    language: Language,
    settings: Settings,
) -> str | None:
    """Run claim-based verification on the answer. Returns improved answer or None."""
    if not settings.llm_verification_enabled:
        return None

    evidence_block = _build_evidence_block(contexts)
    prompt_template = _verification_prompt(language)
    prompt = prompt_template.format(
        evidence=evidence_block[:6000],
        question=question,
        answer=answer,
    )

    try:
        raw = _chat_completion(
            settings=settings,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict fact-checker. Extract every claim and verify against evidence. Return ONLY valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            model_override=settings.llm_reasoning_model,
            max_tokens_override=settings.llm_reasoning_max_tokens,
        )
        parsed = _extract_json_payload(raw)

        has_hallucination = parsed.get("has_hallucination", False)
        issues = parsed.get("issues", [])

        if not has_hallucination:
            logger.info("Answer verification PASSED — no hallucinations detected")
            return None

        unsupported_count = sum(
            1 for c in parsed.get("claims", [])
            if c.get("verdict") == "unsupported"
        )
        logger.warning(
            "Answer verification found %d unsupported claims: %s",
            unsupported_count,
            issues,
        )

        # Prefer the improved answer from verification (strips hallucinations)
        improved = parsed.get("improved_answer")
        if improved and isinstance(improved, str) and improved.strip():
            return improved.strip()

    except Exception as exc:
        logger.warning("Answer verification failed (non-critical): %s", exc)

    return None


# ---------------------------------------------------------------------------
# Query Decomposition — breaks complex questions into sub-queries
# ---------------------------------------------------------------------------


def decompose_query(
    query: str,
    language: Language,
    settings: Settings,
) -> list[str]:
    """Decompose a complex question into sub-queries. Returns empty list if simple."""
    if not settings.llm_query_decomposition_enabled:
        return []

    # Quick heuristic: skip decomposition for short/simple questions
    if len(query.split()) < 8:
        return []

    prompt_template = _decomposition_prompt(language)
    prompt = prompt_template.format(query=query)

    try:
        raw = _chat_completion(
            settings=settings,
            messages=[
                {
                    "role": "system",
                    "content": "You are a query decomposition assistant. Return ONLY valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            model_override=settings.llm_reasoning_model,
            max_tokens_override=1024,
        )
        parsed = _extract_json_payload(raw)
        sub_queries = parsed.get("sub_queries", [])

        if isinstance(sub_queries, list) and len(sub_queries) >= 2:
            logger.info(
                "Query decomposed into %d sub-queries: %s",
                len(sub_queries),
                sub_queries,
            )
            return [sq for sq in sub_queries if isinstance(sq, str) and sq.strip()]

    except Exception as exc:
        logger.warning("Query decomposition failed (non-critical): %s", exc)

    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_answer_stream_validated(
    question: str,
    contexts: list[RetrievedContext],
    language: Language,
    settings: Settings,
    history: list[dict[str, str]] | None = None,
) -> Generator[str, None, None]:
    use_cot = settings.llm_cot_enabled and _is_complex_query(question)
    messages = _build_grounded_messages(
        question=question,
        contexts=contexts,
        language=language,
        history=history,
        use_cot=use_cot,
    )
    buffer = ""
    has_json_structure = False
    for token in _chat_completion_stream(settings, messages):
        buffer += token
        if not has_json_structure and any(c in buffer for c in _JSON_OPENERS):
            has_json_structure = True
        yield token
    if not has_json_structure:
        logger.warning(
            "Streamed response lacks JSON structure for query: %s", question[:100]
        )


def reformulate_query(
    message: str,
    history: list[dict[str, str]] | None,
    settings: Settings,
) -> str:
    if not history:
        return message
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:200]}" for m in (history or [])[-4:]
    )
    try:
        raw = _chat_completion(
            settings=settings,
            messages=[
                {
                    "role": "user",
                    "content": _REFORMULATE_PROMPT.format(
                        history=history_text, message=message
                    ),
                }
            ],
        )
        result = raw.strip() or message
        logger.info("Query reformulated: '%s' -> '%s'", message, result)
        return result
    except Exception as exc:
        logger.warning("Query reformulation failed, using original: %s", exc)
        return message


def generate_query_variations(
    query: str,
    language: Language,
    settings: Settings,
    count: int = 2,
) -> list[str]:
    try:
        raw = _chat_completion(
            settings=settings,
            messages=[
                {
                    "role": "user",
                    "content": _QUERY_VARIATIONS_PROMPT.format(
                        count=count,
                        language=language.value,
                        query=query,
                    ),
                }
            ],
        )
        variations = [line.strip() for line in raw.split("\n") if line.strip()]
        variations = [re.sub(r"^\d+[\.\):]\s*", "", v) for v in variations]
        logger.info("Generated %d variations for query '%s'", len(variations), query)
        return variations[:count]
    except Exception as exc:
        logger.warning("Query variations generation failed: %s", exc)
        return []


def generate_hypothetical_answer(
    query: str,
    settings: Settings,
) -> str:
    prompt = (
        "Write a short factual passage that could answer the question. "
        "This passage will be used only for document retrieval.\n\n"
        f"Question: {query}\n\nPassage:"
    )
    try:
        return (
            _chat_completion(
                settings=settings,
                messages=[{"role": "user", "content": prompt}],
            ).strip()
            or query
        )
    except Exception as exc:
        logger.warning("HyDE generation failed: %s", exc)
        return query


def generate_chitchat_answer(
    question: str,
    language: Language,
    settings: Settings,
    history: list[dict[str, str]] | None = None,
) -> RAGResponse:
    answer = _chat_completion(
        settings=settings,
        messages=_build_chitchat_messages(question, language, history),
    )
    return RAGResponse(
        answer=answer,
        detected_language=language,
        citations=[],
        evidence=[],
        grounding_status=GroundingStatus.GROUNDED,
        supplement_used=False,
        model_used=settings.llm_model,
    )


def generate_answer(
    question: str,
    contexts: list[RetrievedContext],
    language: Language,
    settings: Settings,
    history: list[dict[str, str]] | None = None,
) -> RAGResponse:
    if not contexts:
        answer = _unsupported_message(language)
        return RAGResponse(
            answer=answer,
            detected_language=language,
            citations=[],
            evidence=[],
            grounding_status=GroundingStatus.UNSUPPORTED,
            supplement_used=False,
            model_used=settings.llm_model,
        )

    # Try structured listing for procedure queries first
    structured_listing = _synthesize_procedure_listing_answer(
        question=question,
        contexts=contexts,
        language=language,
        settings=settings,
    )
    if structured_listing is not None:
        return structured_listing

    # Determine if we should use Chain-of-Thought reasoning
    use_cot = settings.llm_cot_enabled and _is_complex_query(question)

    validation_feedback: str | None = None
    last_error: Exception | None = None

    for attempt in range(settings.llm_answer_attempts):
        raw = _chat_completion(
            settings=settings,
            messages=_build_grounded_messages(
                question=question,
                contexts=contexts,
                language=language,
                history=history,
                validation_feedback=validation_feedback,
                use_cot=use_cot,
            ),
        )
        try:
            payload = _extract_grounded_payload(raw)
            payload = _normalize_payload(payload, contexts, language)
            if payload.grounding_status == GroundingStatus.UNSUPPORTED:
                synthesized = _synthesize_listing_answer(
                    question=question,
                    contexts=contexts,
                    language=language,
                    settings=settings,
                )
                if synthesized is not None:
                    return synthesized
                return RAGResponse(
                    answer=_unsupported_message(language),
                    detected_language=language,
                    citations=[],
                    evidence=[],
                    grounding_status=GroundingStatus.UNSUPPORTED,
                    supplement_used=False,
                    model_used=settings.llm_model,
                )
            answer_text, supplement_used = _compose_answer_text(payload, language)

            # Self-verification: check answer quality and improve if needed
            if settings.llm_verification_enabled and answer_text:
                improved = _verify_answer(
                    question=question,
                    answer=answer_text,
                    contexts=contexts,
                    language=language,
                    settings=settings,
                )
                if improved:
                    logger.info("Using self-verified improved answer")
                    # Re-extract citations from improved answer
                    improved_ids = _extract_answer_citation_ids(improved)
                    if improved_ids:
                        answer_text = improved
                        payload = payload.model_copy(
                            update={
                                "answer": improved,
                                "cited_evidence_ids": improved_ids,
                            }
                        )

            # Post-processing: validate citations and flag uncited paragraphs
            answer_text = _validate_citations_in_answer(
                answer_text, contexts, language
            )

            cited_contexts = _filter_cited_contexts(
                contexts,
                payload.cited_evidence_ids,
            )
            return RAGResponse(
                answer=answer_text,
                detected_language=language,
                citations=cited_contexts,
                evidence=contexts,
                grounding_status=payload.grounding_status,
                supplement_used=supplement_used,
                model_used=settings.llm_model,
            )
        except Exception as exc:
            last_error = exc
            validation_feedback = (
                "Ensure the 'answer' field contains the full comprehensive response with [E1] style citations. "
                "The entire output MUST be a single valid JSON object and 'supplement' must be null. "
                "Make the 'answer' field detailed and explanatory."
            )
            logger.warning(
                "Grounded answer validation failed (attempt %d): %s", attempt + 1, exc
            )

    synthesized = _synthesize_listing_answer(
        question=question,
        contexts=contexts,
        language=language,
        settings=settings,
    )
    if synthesized is not None:
        return synthesized

    logger.warning("Falling back to unsupported response: %s", last_error)
    return RAGResponse(
        answer=_unsupported_message(language),
        detected_language=language,
        citations=[],
        evidence=[],
        grounding_status=GroundingStatus.UNSUPPORTED,
        supplement_used=False,
        model_used=settings.llm_model,
    )


def generate_answer_stream(
    question: str,
    contexts: list[RetrievedContext],
    language: Language,
    settings: Settings,
    history: list[dict[str, str]] | None = None,
) -> Generator[str, None, None]:
    if not contexts:
        yield _unsupported_message(language)
        return
    yield from generate_answer_stream_validated(
        question=question,
        contexts=contexts,
        language=language,
        settings=settings,
        history=history,
    )


def generate_answer_plain_stream(
    question: str,
    contexts: list[RetrievedContext],
    language: Language,
    settings: Settings,
    history: list[dict[str, str]] | None = None,
) -> Generator[str, None, None]:
    if not contexts:
        yield _unsupported_message(language)
        return

    system_prompt = (
        "Anda adalah \"I'm in Charge\", asisten dokumen Gapura Airport Services. "
        "Jawab singkat, jelas, dan dalam bahasa pengguna. Gunakan hanya bukti dokumen. "
        "Setiap klaim faktual wajib memakai sitasi seperti [E1]. "
        "Jika bukti tidak cukup, katakan bahwa dokumen tidak cukup mendukung."
        if language == Language.ID
        else "You are \"I'm in Charge\", a Gapura Airport Services document assistant. "
        "Answer briefly and clearly in the user's language. Use only document evidence. "
        "Every factual claim must include citations like [E1]. "
        "If evidence is insufficient, say the documents do not sufficiently support the answer."
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(_build_history_messages(history))
    prompt_contexts = contexts[:3]
    messages.append(
        {
            "role": "user",
            "content": (
                f"Evidence:\n{_build_evidence_block(prompt_contexts)[:2500]}\n\n"
                f"Question: {question}\n\nAnswer:"
            ),
        }
    )

    yield from _chat_completion_stream(
        settings=settings,
        messages=messages,
        max_tokens_override=min(settings.llm_max_tokens, 1024),
    )
