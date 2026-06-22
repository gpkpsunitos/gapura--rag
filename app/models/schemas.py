from typing import Any

from pydantic import BaseModel, Field

from app.models.types import ChunkId, DocId, GroundingStatus, Language


class PageContent(BaseModel):
    page_number: int
    text: str


class DocumentChunk(BaseModel):
    chunk_id: ChunkId
    doc_id: DocId
    text: str
    page: int
    chunk_index: int
    language: Language
    source_filename: str
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    language_override: Language | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    sources: list[str] | None = None


class RetrievedContext(BaseModel):
    evidence_id: str = ""
    text: str
    snippet: str = ""
    score: float
    rerank_score: float | None = None
    source_filename: str
    page: int
    chunk_id: ChunkId
    doc_id: DocId | None = None
    chunk_index: int | None = None


class GroundedAnswerPayload(BaseModel):
    grounding_status: GroundingStatus
    answer: str = ""
    cited_evidence_ids: list[str] = Field(default_factory=list)
    supplement: str | None = None


class RAGResponse(BaseModel):
    answer: str
    detected_language: Language
    citations: list[RetrievedContext] = Field(default_factory=list)
    evidence: list[RetrievedContext] = Field(default_factory=list)
    grounding_status: GroundingStatus = GroundingStatus.UNSUPPORTED
    supplement_used: bool = False
    model_used: str


class IngestionResult(BaseModel):
    doc_id: DocId
    source_filename: str
    total_pages: int
    total_chunks: int
    skipped: bool = False

DocumentChunk.model_rebuild(_types_namespace={"Any": Any})
