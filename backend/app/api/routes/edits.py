from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.api.deps import get_orchestrator
from app.core.models import EditRequest, EditResponse

router = APIRouter()


@router.post("", response_model=EditResponse)
def propose_edit(request: EditRequest) -> EditResponse:
    return _propose(request)


@router.post("/propose", response_model=EditResponse)
def propose_edit_explicit(request: EditRequest) -> EditResponse:
    return _propose(request)


@router.post("/{command_id}/apply", response_model=EditResponse)
def apply_edit(command_id: UUID) -> EditResponse:
    orchestrator = get_orchestrator()
    try:
        return orchestrator.apply_edit(command_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/{command_id}/reject", response_model=EditResponse)
def reject_edit(command_id: UUID) -> EditResponse:
    orchestrator = get_orchestrator()
    try:
        return orchestrator.reject_edit(command_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _propose(request: EditRequest) -> EditResponse:
    orchestrator = get_orchestrator()
    try:
        return orchestrator.propose_edit(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
