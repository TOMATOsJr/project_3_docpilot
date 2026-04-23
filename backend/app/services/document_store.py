from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from uuid import UUID

from app.core.models import Chunk, ChunkMetadata, DocumentMetadata, DocumentType


@dataclass(slots=True)
class StoredDocument:
    metadata: DocumentMetadata
    raw_text: str
    chunks: list[Chunk] = field(default_factory=list)


class InMemoryDocumentStore:
    def __init__(self) -> None:
        self._documents: dict[UUID, StoredDocument] = {}

    def list_documents(self) -> list[DocumentMetadata]:
        return [record.metadata for record in self._documents.values()]

    def get_document(self, document_id: UUID) -> StoredDocument:
        return self._documents[document_id]

    def save_uploaded_file(self, filename: str, content: bytes) -> StoredDocument:
        document_type = _detect_document_type(filename)
        text = _extract_text(filename, content)
        metadata = DocumentMetadata(filename=filename, document_type=document_type, page_count=1 if text else 0)
        chunks = _chunk_text(metadata.id, text)
        stored = StoredDocument(metadata=metadata, raw_text=text, chunks=chunks)
        self._documents[metadata.id] = stored
        return stored

    def add_document(self, metadata: DocumentMetadata, raw_text: str, chunks: Iterable[Chunk]) -> StoredDocument:
        stored = StoredDocument(metadata=metadata, raw_text=raw_text, chunks=list(chunks))
        self._documents[metadata.id] = stored
        return stored

    def delete_document(self, document_id: UUID) -> None:
        self._documents.pop(document_id, None)


def _detect_document_type(filename: str) -> DocumentType:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return DocumentType.pdf
    if suffix == ".docx":
        return DocumentType.docx
    if suffix == ".pptx":
        return DocumentType.pptx
    if suffix in {".md", ".markdown"}:
        return DocumentType.markdown
    if suffix in {".txt", ""}:
        return DocumentType.text
    return DocumentType.unknown


def _extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".md", ".markdown", ".txt", ""}:
        return content.decode("utf-8", errors="ignore")
    return content.decode("utf-8", errors="ignore")


def _chunk_text(document_id: UUID, text: str, chunk_size: int = 800) -> list[Chunk]:
    if not text.strip():
        return []

    paragraphs = [paragraph.strip() for paragraph in text.splitlines() if paragraph.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: list[Chunk] = []
    page_number = 1
    for paragraph_index, paragraph in enumerate(paragraphs, start=1):
        if len(paragraph) <= chunk_size:
            chunks.append(
                Chunk(
                    text=paragraph,
                    metadata=ChunkMetadata(
                        document_id=document_id,
                        page_number=page_number,
                        paragraph_index=paragraph_index,
                        source_label=f"p{page_number}-para{paragraph_index}",
                    ),
                )
            )
            continue

        start = 0
        part_index = 0
        while start < len(paragraph):
            part_index += 1
            piece = paragraph[start : start + chunk_size]
            chunks.append(
                Chunk(
                    text=piece,
                    metadata=ChunkMetadata(
                        document_id=document_id,
                        page_number=page_number,
                        paragraph_index=paragraph_index,
                        source_label=f"p{page_number}-para{paragraph_index}-part{part_index}",
                    ),
                )
            )
            start += chunk_size
        page_number += 1
    return chunks
