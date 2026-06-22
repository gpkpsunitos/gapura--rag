from __future__ import annotations

import hashlib
from enum import Enum
from typing import NewType

EmbeddingVector = NewType("EmbeddingVector", list[float])
ChunkId = NewType("ChunkId", str)
DocId = NewType("DocId", str)


class Language(str, Enum):
    EN = "en"
    ID = "id"


class GroundingStatus(str, Enum):
    GROUNDED = "grounded"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


# Complexity: Time O(n) | Space O(1) — streams file bytes through SHA-256
def compute_doc_id(file_bytes: bytes) -> DocId:
    return DocId(hashlib.sha256(file_bytes).hexdigest())


# Complexity: Time O(1) | Space O(1)
def build_chunk_id(doc_id: DocId, page: int, index: int) -> ChunkId:
    return ChunkId(f"{doc_id[:16]}_{page}_{index}")
