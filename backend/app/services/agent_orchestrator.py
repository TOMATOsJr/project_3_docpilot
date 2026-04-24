from __future__ import annotations

from uuid import UUID

from app.core.abstractions import DocumentRepository
from app.core.models import EditRequest, EditResponse, QueryRequest, SynthesisRequest
from app.services.agent_strategies import EditAgentStrategy, PendingEdit, QAAgentStrategy
from app.services.model_gateway import ModelGateway
from app.services.rag_engine import RagEngine


class AgentOrchestrator:
    def __init__(self, rag_engine: RagEngine, model_gateway: ModelGateway, repository: DocumentRepository) -> None:
        self._model_gateway = model_gateway
        self._qa_agent = QAAgentStrategy(rag_engine, model_gateway)
        self._edit_agent = EditAgentStrategy(repository, model_gateway)
        self._pending_edits: dict[UUID, PendingEdit] = {}

    def answer_question(self, request: QueryRequest):
        return self._qa_agent.execute(request)

    def propose_edit(self, request: EditRequest) -> EditResponse:
        pending = self._edit_agent.propose(request)
        self._pending_edits[pending.proposal.command_id] = pending
        return EditResponse(
            proposal=pending.proposal,
            model_used=pending.model_used,
            model_selection_reason=pending.model_selection_reason,
            fallback_used=pending.fallback_used,
            status="pending",
        )

    def apply_edit(self, command_id: UUID):
        pending = self._pending_edits.get(command_id)
        if pending is None:
            raise ValueError("Unknown edit proposal")
        return self._edit_agent.apply(pending)

    def reject_edit(self, command_id: UUID):
        pending = self._pending_edits.get(command_id)
        if pending is None:
            raise ValueError("Unknown edit proposal")
        return self._edit_agent.reject(pending)

    def synthesize(self, request: SynthesisRequest):
        requested_model = getattr(request, 'requested_model', None)
        response, _model_used, _fallback_used, _selection_reason = self._model_gateway.complete(
            request.query,
            task_type="synthesis",
            requested_model=requested_model
        )
        return response
