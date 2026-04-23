from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import get_repository, get_ingest_service
from app.core.models import UploadResponse
from app.infrastructure.persistence import PostgresRepository
from app.services.ingest_service import IngestService

router = APIRouter()


@router.get("", response_model=list[UploadResponse])
def list_documents(repository: PostgresRepository = Depends(get_repository)) -> list[UploadResponse]:
    """List all uploaded documents."""
    documents = repository.list_documents()
    result = []
    for doc in documents:
        chunks = repository.get_chunks_by_document(doc.id)
        result.append(UploadResponse(document=doc, chunk_count=len(chunks)))
    return result


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    ingest_service: IngestService = Depends(get_ingest_service),
) -> UploadResponse:
    """Upload and ingest a document."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a name")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    try:
        # Ingest document using paragraph-based chunking
        document_id = ingest_service.ingest(file.filename, content, chunking_strategy="paragraph")

        # Retrieve stored document for response
        result = ingest_service.repository.get_document_by_id(document_id)
        if not result:
            raise HTTPException(status_code=500, detail="Failed to retrieve stored document")

        metadata, chunks = result
        return UploadResponse(document=metadata, chunk_count=len(chunks))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(error)}") from error


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: UUID,
    repository: PostgresRepository = Depends(get_repository),
) -> None:
    """Delete a document and all its chunks."""
    try:
        # Check if document exists
        result = repository.get_document_by_id(document_id)
        if not result:
            raise HTTPException(status_code=404, detail="Document not found")

        repository.delete_document(document_id)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(error)}") from error
