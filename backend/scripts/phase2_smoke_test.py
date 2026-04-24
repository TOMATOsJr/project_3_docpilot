from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from uuid import UUID

# Ensure backend package import works when running from repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.abstractions import DocumentRepository, StoredChunk
from app.core.models import Chunk, DocumentMetadata
from app.services.ingest_service import IngestService


class InMemoryTestRepository(DocumentRepository):
    def __init__(self) -> None:
        self._docs: dict[UUID, tuple[DocumentMetadata, str, list[Chunk]]] = {}

    def save_document(self, metadata: DocumentMetadata, raw_text: str, chunks: list[Chunk]) -> UUID:
        self._docs[metadata.id] = (metadata, raw_text, chunks)
        return metadata.id

    def get_document_by_id(self, document_id: UUID) -> Optional[tuple[DocumentMetadata, list[Chunk]]]:
        record = self._docs.get(document_id)
        if record is None:
            return None
        metadata, _raw_text, chunks = record
        return metadata, chunks

    def get_document_text(self, document_id: UUID) -> Optional[tuple[DocumentMetadata, str]]:
        record = self._docs.get(document_id)
        if record is None:
            return None
        metadata, raw_text, _chunks = record
        return metadata, raw_text

    def list_documents(self) -> list[DocumentMetadata]:
        return [record[0] for record in self._docs.values()]

    def delete_document(self, document_id: UUID) -> None:
        self._docs.pop(document_id, None)

    def get_chunks_by_document(self, document_id: UUID) -> list[Chunk]:
        record = self._docs.get(document_id)
        return [] if record is None else record[2]

    def replace_document_content(self, document_id: UUID, raw_text: str, chunks: list[Chunk]) -> None:
        record = self._docs.get(document_id)
        if record is None:
            return
        metadata, _existing_text, _existing_chunks = record
        self._docs[document_id] = (metadata, raw_text, chunks)

    def save_chunk_embedding(self, chunk_id: UUID, embedding: list[float]) -> None:
        return

    def search_chunks_by_embedding(self, query_embedding: list[float], top_k: int = 5) -> list[StoredChunk]:
        return []

    def search_chunks_by_keyword(
        self,
        keywords: list[str],
        document_id: Optional[UUID] = None,
        top_k: int = 5,
    ) -> list[StoredChunk]:
        return []


def main() -> int:
    repo = InMemoryTestRepository()
    ingest = IngestService(repository=repo)

    filename = "phase2-smoke.md"
    content = b"# Title\n\nDocPilot phase 2 smoke test content.\n\nSecond paragraph for chunking."

    doc_id = ingest.ingest(filename=filename, content=content, chunking_strategy="paragraph")
    result = repo.get_document_by_id(doc_id)

    if result is None:
        print("FAIL: document was not stored")
        return 1

    metadata, chunks = result
    if metadata.filename != filename:
        print("FAIL: metadata filename mismatch")
        return 1

    if not chunks:
        print("FAIL: no chunks created")
        return 1

    print(f"PASS: document={metadata.id} chunks={len(chunks)} type={metadata.document_type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
