from __future__ import annotations

from app.core.models import EditRequest, QueryRequest, SynthesisRequest
from app.services.edit_manager import EditCommand
from app.services.model_gateway import ModelGateway
from app.services.rag_engine import RagEngine


class AgentOrchestrator:
    def __init__(self, rag_engine: RagEngine, model_gateway: ModelGateway) -> None:
        self._rag_engine = rag_engine
        self._model_gateway = model_gateway

    def answer_question(self, request: QueryRequest):
        response = self._rag_engine.answer(request)
        # If we have a model_gateway, use it to refine the model_used field
        if self._model_gateway and request.requested_model:
            is_valid = self._model_gateway.validate_model(request.requested_model)
            response.requested_model = request.requested_model if is_valid else None
        return response

    def propose_edit(self, request: EditRequest):
        original_text = request.selected_text or ""
        # Pass requested_model to model_gateway if available
        requested_model = getattr(request, 'requested_model', None)
        proposed_text, model_used, fallback_used = self._model_gateway.complete(
            request.instruction,
            task_type="edit",
            requested_model=requested_model
        )
        command = EditCommand(
            document_id=request.document_id,
            instruction=request.instruction,
            original_text=original_text,
            proposed_text=proposed_text,
        )
        return command.to_proposal()

    def synthesize(self, request: SynthesisRequest):
        requested_model = getattr(request, 'requested_model', None)
        response, model_used, fallback_used = self._model_gateway.complete(
            request.query,
            task_type="synthesis",
            requested_model=requested_model
        )
        return response
