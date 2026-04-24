from functools import lru_cache

from app.config import get_settings
from app.services.agent_orchestrator import AgentOrchestrator
from app.services.model_gateway import ModelGateway
from app.services.rag_engine import RagEngine
from app.services.ingest_service import IngestService
from app.infrastructure.persistence import PostgresRepository


@lru_cache(maxsize=1)
def get_repository() -> PostgresRepository:
    """Get PostgreSQL repository for document persistence."""
    settings = get_settings()
    return PostgresRepository(settings.database_url, echo=False)


@lru_cache(maxsize=1)
def get_ingest_service() -> IngestService:
    """Get document ingestion service."""
    return IngestService(repository=get_repository())


@lru_cache(maxsize=1)
def get_model_gateway() -> ModelGateway:
    settings = get_settings()
    return ModelGateway(
        primary_model=settings.primary_model,
        fallback_model=settings.fallback_model,
        allowed_models=settings.allowed_models,
        anthropic_api_key=settings.anthropic_api_key,
        openai_api_key=settings.openai_api_key,
        gemini_api_key=settings.gemini_api_key,
        groq_api_key=settings.groq_api_key,
    )


@lru_cache(maxsize=1)
def get_rag_engine() -> RagEngine:
    """Get RAG engine using PostgreSQL repository and ModelGateway."""
    return RagEngine(repository=get_repository(), model_gateway=get_model_gateway())


@lru_cache(maxsize=1)
def get_orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator(get_rag_engine(), get_model_gateway(), get_repository())
