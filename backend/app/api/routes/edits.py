from fastapi import APIRouter

from app.api.deps import get_orchestrator
from app.core.models import EditRequest, EditResponse

router = APIRouter()


@router.post("", response_model=EditResponse)
def propose_edit(request: EditRequest) -> EditResponse:
    orchestrator = get_orchestrator()
    proposal = orchestrator.propose_edit(request)
    return EditResponse(proposal=proposal)
