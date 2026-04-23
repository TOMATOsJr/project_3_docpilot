from __future__ import annotations

from uuid import UUID
from typing import Optional

from app.core.models import Chunk, Citation, QueryRequest, QueryResponse
from app.core.abstractions import DocumentRepository


class RagEngine:
    """RAG Engine using DocumentRepository for retrieval."""

    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    def answer(self, request: QueryRequest) -> QueryResponse:
        """Answer a query using retrieval-augmented generation."""
        retrieved_chunks = self._retrieve_chunks(
            request.query,
            document_ids=request.document_ids,
            top_k=request.max_chunks,
        )
        answer = self._compose_answer(request.query, retrieved_chunks)
        citations = [
            Citation(
                document_id=chunk.metadata.document_id,
                page_number=chunk.metadata.page_number,
                paragraph_index=chunk.metadata.paragraph_index,
                source_label=chunk.metadata.source_label or f"Chunk {chunk.metadata.paragraph_index}",
                quote=chunk.text[:200],
            )
            for chunk in retrieved_chunks
        ]
        return QueryResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            model_used="keyword-retrieval",
        )

    def _retrieve_chunks(
        self,
        query: str,
        document_ids: Optional[list[UUID]] = None,
        top_k: int = 5,
    ) -> list[Chunk]:
        """Retrieve chunks matching query using keyword search."""
        query_terms = [term.lower() for term in query.split() if len(term) > 2]

        # Search by keyword for each document (or all if none specified)
        results = []
        if document_ids:
            for doc_id in document_ids:
                chunks = self._repository.get_chunks_by_document(doc_id)
                results.extend(chunks)
        else:
            # Get all documents and their chunks
            all_docs = self._repository.list_documents()
            for doc in all_docs:
                chunks = self._repository.get_chunks_by_document(doc.id)
                results.extend(chunks)

        # Score and sort chunks
        scored_chunks = [(chunk, self._score_chunk(chunk.text, query_terms)) for chunk in results]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return [chunk for chunk, score in scored_chunks if score > 0][:top_k]

    def _score_chunk(self, text: str, query_terms: list[str]) -> int:
        """Score a chunk by keyword overlap."""
        lowered = text.lower()
        return sum(1 for term in query_terms if term in lowered)

    def _compose_answer(self, query: str, chunks: list[Chunk]) -> str:
        """Compose a grounded answer from retrieved chunks."""
        if not chunks:
            return f"I could not find grounded evidence for: {query}"

        evidence = "; ".join(chunk.text[:120].replace("\n", " ") for chunk in chunks)
        return f"Based on the uploaded documents, {query.strip().rstrip('?')}. Evidence: {evidence}"
