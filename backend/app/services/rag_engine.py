from __future__ import annotations

from uuid import UUID
from typing import Optional

from app.core.models import Chunk, Citation, QueryRequest, QueryResponse
from app.core.abstractions import DocumentRepository
from app.services.model_gateway import ModelGateway


class RagEngine:
    """RAG Engine using DocumentRepository for retrieval and ModelGateway for answer synthesis."""

    def __init__(self, repository: DocumentRepository, model_gateway: ModelGateway | None = None) -> None:
        self._repository = repository
        self._model_gateway = model_gateway

    def answer(self, request: QueryRequest) -> QueryResponse:
        """Answer a query using retrieval-augmented generation."""
        retrieved_chunks = self._retrieve_chunks(
            request.query,
            document_ids=request.document_ids,
            top_k=request.max_chunks,
        )
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

        # If no evidence is retrieved, return a grounded failure instead of hallucinating.
        if not retrieved_chunks:
            answer = f"I could not find grounded evidence for: {request.query}"
            model_used = "keyword-retrieval"
            fallback_used = False
            return QueryResponse(
                answer=answer,
                citations=citations,
                retrieved_chunks=retrieved_chunks,
                model_used=model_used,
                requested_model=request.requested_model,
                fallback_used=fallback_used,
            )

        # Use selected LLM to synthesize answer from retrieved evidence.
        if self._model_gateway is not None:
            grounded_prompt = self._build_grounded_prompt(
                query=request.query,
                chunks=retrieved_chunks,
                conversation_history=request.conversation_history,
            )
            answer, model_used, fallback_used = self._model_gateway.complete(
                grounded_prompt,
                task_type="qa",
                requested_model=request.requested_model,
            )
        else:
            answer = self._compose_answer(
                request.query,
                retrieved_chunks,
                conversation_history=request.conversation_history,
            )
            model_used = "keyword-retrieval"
            fallback_used = False

        return QueryResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            model_used=model_used,
            requested_model=request.requested_model,
            fallback_used=fallback_used,
        )

    def _build_grounded_prompt(self, query: str, chunks: list[Chunk], conversation_history: list | None = None) -> str:
        """Build a strict grounded prompt for LLM synthesis from retrieved evidence."""
        evidence_lines = []
        for idx, chunk in enumerate(chunks, start=1):
            src = chunk.metadata.source_label or f"chunk-{idx}"
            page = chunk.metadata.page_number if chunk.metadata.page_number is not None else "n/a"
            para = chunk.metadata.paragraph_index if chunk.metadata.paragraph_index is not None else "n/a"
            evidence_lines.append(f"[{idx}] source={src}; page={page}; paragraph={para}; text={chunk.text}")

        history_block = ""
        if conversation_history:
            turns = [f"User: {turn.query}\nAssistant: {turn.answer}" for turn in conversation_history]
            history_block = "\n\nConversation History:\n" + "\n".join(turns)

        return (
            "You are a grounded assistant. Answer only using the provided evidence snippets. "
            "If evidence is insufficient, say that clearly. Do not invent facts.\n\n"
            f"User Question:\n{query}\n"
            f"{history_block}\n\n"
            "Evidence Snippets:\n"
            + "\n".join(evidence_lines)
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

    def _compose_answer(self, query: str, chunks: list[Chunk], conversation_history: list | None = None) -> str:
        """Compose a grounded answer from retrieved chunks and conversation history."""
        if not chunks:
            return f"I could not find grounded evidence for: {query}"

        evidence = "; ".join(chunk.text[:120].replace("\n", " ") for chunk in chunks)

        # Build context string from history if provided
        context_str = ""
        if conversation_history:
            context_parts = [f"User: {turn.query}\nAssistant: {turn.answer}" for turn in conversation_history]
            context_str = "\n\nPrior conversation:\n" + "\n".join(context_parts) + "\n\n"

        return f"Based on the uploaded documents,{context_str} {query.strip().rstrip('?')}. Evidence: {evidence}"
