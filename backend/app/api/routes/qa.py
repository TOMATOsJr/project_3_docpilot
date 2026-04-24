from fastapi import APIRouter

from app.api.deps import get_orchestrator, get_model_gateway
from app.core.models import QueryRequest, QueryResponse

router = APIRouter()


@router.post("", response_model=QueryResponse)
def ask_question(request: QueryRequest) -> QueryResponse:
    orchestrator = get_orchestrator()
    return orchestrator.answer_question(request)


@router.get("/models")
def get_available_models() -> dict[str, list[str]]:
    """Get list of available models for frontend selection."""
    model_gateway = get_model_gateway()
    return {"available_models": model_gateway.get_allowed_models()}
