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
        return self._rag_engine.answer(request)

    def propose_edit(self, request: EditRequest):
        original_text = request.selected_text or ""
        proposed_text = self._model_gateway.complete(request.instruction, task_type="edit")
        command = EditCommand(
            document_id=request.document_id,
            instruction=request.instruction,
            original_text=original_text,
            proposed_text=proposed_text,
        )
        return command.to_proposal()

    def synthesize(self, request: SynthesisRequest):
        response = self._model_gateway.complete(request.query, task_type="synthesis")
        return response
