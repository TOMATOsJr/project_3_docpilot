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
        # Ensure we request enough chunks to potentially fill the context window, instead of strict top_k
        raw_chunks = self._retrieve_chunks(
            request.query,
            document_ids=request.document_ids,
            top_k=100,
        )

        # Determine model
        model = request.requested_model
        if self._model_gateway:
            model = self._model_gateway.select_model(
                task_type="qa",
                prompt=request.query,
                requested_model=request.requested_model,
            )

        # Apply budget allocation (100% chunks first, then history if space remains)
        retrieved_chunks, filtered_history = self._apply_context_budget(
            request.query, raw_chunks, request.conversation_history, model or "gemini/gemini-2.5-flash-lite"
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

        # If no evidence is retrieved and no history is present, return a grounded failure instead of hallucinating.
        if not retrieved_chunks and not request.conversation_history:
            answer = f"I could not find grounded evidence for: {request.query}"
            model_used = "keyword-retrieval"
            fallback_used = False
            model_selection_reason = "No evidence available; returned retrieval-only response without LLM synthesis."
            return QueryResponse(
                answer=answer,
                citations=citations,
                retrieved_chunks=retrieved_chunks,
                model_used=model_used,
                model_selection_reason=model_selection_reason,
                requested_model=request.requested_model,
                fallback_used=fallback_used,
            )

        # Use selected LLM to synthesize answer from retrieved evidence.
        if self._model_gateway is not None:
            grounded_prompt = self._build_grounded_prompt(
                query=request.query,
                chunks=retrieved_chunks,
                conversation_history=filtered_history,
            )
            answer, model_used, fallback_used, model_selection_reason = self._model_gateway.complete(
                grounded_prompt,
                task_type="qa",
                requested_model=request.requested_model,
            )
        else:
            answer = self._compose_answer(
                request.query,
                retrieved_chunks,
                conversation_history=filtered_history,
            )
            model_used = "keyword-retrieval"
            fallback_used = False
            model_selection_reason = "Model gateway disabled; used retrieval-only answer composition."

        return QueryResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            model_used=model_used,
            model_selection_reason=model_selection_reason,
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

        evidence_section = "Evidence Snippets:\n" + "\n".join(evidence_lines) if chunks else "Evidence Snippets:\n[No reference present. You may answer from the Conversation History.]"

        return (
            "You are a grounded assistant. Answer only using the provided evidence snippets and the conversation history. "
            "If evidence is insufficient, say that clearly. Do not invent facts.\n\n"
            f"User Question:\n{query}\n"
            f"{history_block}\n\n"
            f"{evidence_section}"
        )

    def _retrieve_chunks(
        self,
        query: str,
        document_ids: Optional[list[UUID]] = None,
        top_k: int = 100,
    ) -> list[Chunk]:
        """Retrieve chunks matching query using semantic search (cosine similarity > 0.6)."""
        import litellm
        from app.config import get_settings
        import logging

        logger = logging.getLogger(__name__)
        settings = get_settings()

        try:
            response = litellm.embedding(model=settings.embedding_model, input=[query])
            query_embedding = response.data[0]['embedding']
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []

        stored_chunks = self._repository.search_chunks_by_embedding(
            query_embedding, top_k=top_k, document_ids=document_ids
        )

        return [
            Chunk(
                id=sc.id,
                text=sc.text,
                metadata=sc.metadata,
            ) for sc in stored_chunks
        ]

    def _apply_context_budget(
        self,
        query: str,
        chunks: list[Chunk],
        conversation_history: list,
        model: str
    ) -> tuple[list[Chunk], list]:
        """
        Fits chunks and history into 90% of the model's context window.
        Priority: Base Prompt -> All possible Chunks -> Leftover space for History.
        """
        import litellm

        try:
            # Fallback for models without explicit config
            max_tokens = litellm.get_max_tokens(model)
            print("max tokens:",max_tokens)
        except Exception:
            print("Error getting max tokens from litellm, using default value of 8192")
            max_tokens = 8192

        budget = int(0.9 * max_tokens)

        base_prompt = "You are a grounded assistant. Answer only using the provided evidence snippets and the conversation history. If evidence is insufficient, say that clearly. Do not invent facts.\n\nUser Question:\n\n\nEvidence Snippets:\n"

        try:
            used_tokens = litellm.token_counter(model=model, text=base_prompt + query)
        except Exception:
            used_tokens = len(base_prompt + query) // 4

        remaining_budget = budget - used_tokens

        selected_chunks = []
        for chunk in chunks:
            chunk_text = f"[X] source={chunk.metadata.source_label}; page={chunk.metadata.page_number}; paragraph={chunk.metadata.paragraph_index}; text={chunk.text}\n"
            try:
                chunk_tokens = litellm.token_counter(model=model, text=chunk_text)
            except Exception:
                chunk_tokens = len(chunk_text) // 4

            if remaining_budget - chunk_tokens > 0:
                selected_chunks.append(chunk)
                remaining_budget -= chunk_tokens
            else:
                break

        selected_history = []
        for turn in reversed(conversation_history):
            turn_text = f"User: {turn.query}\nAssistant: {turn.answer}\n"
            try:
                turn_tokens = litellm.token_counter(model=model, text=turn_text)
            except Exception:
                turn_tokens = len(turn_text) // 4

            if remaining_budget - turn_tokens > 0:
                selected_history.insert(0, turn)
                remaining_budget -= turn_tokens
            else:
                break

        return selected_chunks, selected_history

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
