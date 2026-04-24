from __future__ import annotations

from dataclasses import dataclass
import logging

import litellm

from app.core.abstractions import DocumentRepository
from app.core.models import DocumentType, EditProposal, EditRequest, EditResponse, QueryRequest
from app.services.edit_manager import EditCommand
from app.services.model_gateway import ModelGateway
from app.services.rag_engine import RagEngine
from app.core.strategies import ChunkingStrategyRegistry


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PendingEdit:
    proposal: EditProposal
    full_text_before: str
    selected_text: str | None
    model_used: str | None
    model_selection_reason: str | None
    fallback_used: bool
    status: str = "pending"


class QAAgentStrategy:
    """Strategy agent for grounded Q&A requests."""

    def __init__(self, rag_engine: RagEngine, model_gateway: ModelGateway) -> None:
        self._rag_engine = rag_engine
        self._model_gateway = model_gateway

    def execute(self, request: QueryRequest):
        response = self._rag_engine.answer(request)
        if request.requested_model:
            is_valid = self._model_gateway.validate_model(request.requested_model)
            response.requested_model = request.requested_model if is_valid else None
        return response


class EditAgentStrategy:
    """Strategy agent for markdown-only edit proposals and resolution."""

    def __init__(self, repository: DocumentRepository, model_gateway: ModelGateway) -> None:
        self._repository = repository
        self._model_gateway = model_gateway

    def propose(self, request: EditRequest) -> PendingEdit:
        metadata_and_text = self._repository.get_document_text(request.document_id)
        if metadata_and_text is None:
            raise ValueError("Document not found")

        metadata, full_text_before = metadata_and_text
        if metadata.document_type != DocumentType.markdown:
            raise ValueError("Edit agent currently supports markdown documents only")

        selected_text = request.selected_text.strip() if request.selected_text else None
        original_text = selected_text if selected_text else full_text_before

        if selected_text and selected_text not in full_text_before:
            raise ValueError("Selected text was not found in the markdown document")

        prompt = _build_edit_prompt(
            instruction=request.instruction,
            source_text=original_text,
            selected_only=selected_text is not None,
        )

        proposed_text, model_used, fallback_used, model_selection_reason = self._model_gateway.complete(
            prompt,
            task_type="edit",
            requested_model=request.requested_model,
        )

        command = EditCommand(
            document_id=request.document_id,
            instruction=request.instruction,
            original_text=original_text,
            proposed_text=proposed_text,
        )
        proposal = command.to_proposal()

        return PendingEdit(
            proposal=proposal,
            full_text_before=full_text_before,
            selected_text=selected_text,
            model_used=model_used,
            model_selection_reason=model_selection_reason,
            fallback_used=fallback_used,
            status="pending",
        )

    def apply(self, pending: PendingEdit) -> EditResponse:
        if pending.status != "pending":
            raise ValueError("Only pending edit proposals can be applied")

        proposal = pending.proposal
        updated_text = _build_applied_document_text(pending)

        strategy = ChunkingStrategyRegistry.get_strategy("paragraph")
        chunks = strategy.chunk(proposal.document_id, updated_text)
        for index, chunk in enumerate(chunks, start=1):
            chunk.metadata.paragraph_index = index

        self._repository.replace_document_content(proposal.document_id, updated_text, chunks)
        self._embed_and_save_chunks(chunks)

        pending.status = "applied"
        return EditResponse(
            proposal=proposal,
            model_used=pending.model_used,
            model_selection_reason=pending.model_selection_reason,
            fallback_used=pending.fallback_used,
            status="applied",
        )

    def reject(self, pending: PendingEdit) -> EditResponse:
        if pending.status != "pending":
            raise ValueError("Only pending edit proposals can be rejected")

        pending.status = "rejected"
        return EditResponse(
            proposal=pending.proposal,
            model_used=pending.model_used,
            model_selection_reason=pending.model_selection_reason,
            fallback_used=pending.fallback_used,
            status="rejected",
        )

    def _embed_and_save_chunks(self, chunks) -> None:
        if not chunks:
            return

        from app.config import get_settings

        settings = get_settings()
        try:
            response = litellm.embedding(model=settings.embedding_model, input=[chunk.text for chunk in chunks])
            for i, chunk in enumerate(chunks):
                if i < len(response.data):
                    embedding = response.data[i]["embedding"]
                    self._repository.save_chunk_embedding(chunk.id, embedding)
        except Exception as error:
            logger.error(f"Failed to regenerate embeddings after markdown apply: {error}")


def _build_edit_prompt(instruction: str, source_text: str, selected_only: bool) -> str:
    scope = "selected snippet" if selected_only else "full markdown document"
    return (
        "You are a precise markdown editor. Apply the user instruction to the provided text. "
        "Preserve valid markdown and return only the rewritten markdown text with no explanation.\n\n"
        f"Edit scope: {scope}\n"
        f"Instruction:\n{instruction}\n\n"
        f"Source markdown:\n{source_text}"
    )


def _build_applied_document_text(pending: PendingEdit) -> str:
    proposal = pending.proposal
    proposed_text = proposal.proposed_text or ""
    original_text = proposal.original_text or ""

    if pending.selected_text is None:
        return proposed_text

    if not original_text:
        raise ValueError("Missing original selected text for snippet apply")

    if original_text not in pending.full_text_before:
        raise ValueError("Original selected text no longer exists in document")

    return pending.full_text_before.replace(original_text, proposed_text, 1)
