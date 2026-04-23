from fastapi import APIRouter

from app.api.deps import get_orchestrator
from app.core.models import QueryRequest, QueryResponse

router = APIRouter()


@router.post("", response_model=QueryResponse)
def ask_question(request: QueryRequest) -> QueryResponse:
    orchestrator = get_orchestrator()
    return orchestrator.answer_question(request)
