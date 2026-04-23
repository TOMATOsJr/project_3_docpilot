from fastapi import APIRouter

from app.api.deps import get_orchestrator
from app.core.models import SynthesisRequest, SynthesisResponse

router = APIRouter()


@router.post("", response_model=SynthesisResponse)
def synthesize(request: SynthesisRequest) -> SynthesisResponse:
    orchestrator = get_orchestrator()
    answer = orchestrator.synthesize(request)
    return SynthesisResponse(answer=answer, document_ids=request.document_ids)
