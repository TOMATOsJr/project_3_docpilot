from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    pdf = "pdf"
    docx = "docx"
    pptx = "pptx"
    markdown = "markdown"
    text = "text"
    unknown = "unknown"


class DocumentMetadata(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    filename: str
    document_type: DocumentType = DocumentType.unknown
    page_count: int = 0
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChunkMetadata(BaseModel):
    document_id: UUID
    page_number: int | None = None
    paragraph_index: int | None = None
    source_label: str | None = None


class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    text: str
    metadata: ChunkMetadata
    score: float = 0.0


class Citation(BaseModel):
    document_id: UUID
    page_number: int | None = None
    paragraph_index: int | None = None
    source_label: str | None = None
    quote: str | None = None


class UploadedDocument(BaseModel):
    document: DocumentMetadata
    chunk_count: int


class UploadResponse(BaseModel):
    document: DocumentMetadata
    chunk_count: int


class ConversationTurn(BaseModel):
    """Single turn in conversation history."""
    query: str
    model_used: str
    answer: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    document_ids: list[UUID] = Field(default_factory=list)
    max_chunks: int = 3
    requested_model: str | None = None
    conversation_history: list[ConversationTurn] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[Chunk] = Field(default_factory=list)
    model_used: str
    model_selection_reason: str | None = None
    requested_model: str | None = None
    fallback_used: bool = False


class EditRequest(BaseModel):
    document_id: UUID
    instruction: str
    selected_text: str | None = None
    requested_model: str | None = None


class DiffLine(BaseModel):
    kind: Literal["insert", "delete", "equal"]
    content: str


class EditProposal(BaseModel):
    command_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    instruction: str
    diff: list[DiffLine] = Field(default_factory=list)
    original_text: str | None = None
    proposed_text: str | None = None


class EditResponse(BaseModel):
    proposal: EditProposal
    model_used: str | None = None
    model_selection_reason: str | None = None
    fallback_used: bool = False
    status: Literal["pending", "applied", "rejected"] = "pending"


class EditResolutionResponse(BaseModel):
    proposal: EditProposal
    status: Literal["applied", "rejected"]


class SynthesisRequest(BaseModel):
    query: str
    document_ids: list[UUID] = Field(default_factory=list)
    requested_model: str | None = None


class SynthesisResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    document_ids: list[UUID] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorResponse(BaseModel):
    detail: str
    context: dict[str, Any] | None = None
