from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import get_model_gateway
from app.api.routes import documents, edits, health, qa, synthesis
from app.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.app_name, version=settings.app_version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(qa.router, prefix="/api/qa", tags=["qa"])
app.include_router(edits.router, prefix="/api/edits", tags=["edits"])
app.include_router(synthesis.router, prefix="/api/synthesis", tags=["synthesis"])


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "status": "ok",
        "version": settings.app_version,
    }


@app.on_event("startup")
def startup_event() -> None:
    get_model_gateway()
