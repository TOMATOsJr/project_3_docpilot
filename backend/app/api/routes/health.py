from fastapi import APIRouter

from app.api.deps import get_model_gateway
from app.config import get_settings
from app.core.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    get_model_gateway()
    return HealthResponse(
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )
