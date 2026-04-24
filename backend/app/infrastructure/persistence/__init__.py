"""
Phase 2: Database Models and PostgreSQL Repository

SQLAlchemy ORM models for persistent storage with pgvector support.
Implements the DocumentRepository abstraction for PostgreSQL backend.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import Column, String, Text, Float, DateTime, ForeignKey, create_engine
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base, Session, sessionmaker
from app.core.abstractions import DocumentRepository, StoredChunk
from app.core.models import Chunk, ChunkMetadata, DocumentMetadata

logger = logging.getLogger(__name__)

Base = declarative_base()


# ============================================================================
# SQLAlchemy ORM Models
# ============================================================================


class DocumentRecord(Base):
    """ORM model for documents."""

    __tablename__ = "documents"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    filename = Column(String(255), nullable=False)
    document_type = Column(String(50), nullable=False)
    raw_text = Column(Text, nullable=False)
    page_count = Column(Float, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChunkRecord(Base):
    """ORM model for chunks with vector support."""

    __tablename__ = "chunks"

    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    document_id = Column(PG_UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Float, nullable=False)
    offset = Column(Float, nullable=False)
    length = Column(Float, nullable=False)
    page_num = Column(Float, nullable=False)
    section_name = Column(String(255), nullable=True)
    text = Column(Text, nullable=False)
    embedding = Column(Text, nullable=True)  # Stored as JSON string for now
    created_at = Column(DateTime, default=datetime.utcnow)


# ============================================================================
# PostgreSQL Repository Implementation
# ============================================================================


class PostgresRepository(DocumentRepository):
    """
    Concrete implementation of DocumentRepository using PostgreSQL + pgvector.
    Handles persistence for documents, chunks, embeddings, and retrieval.
    """

    def __init__(self, database_url: str, echo: bool = False):
        """
        Initialize PostgreSQL repository.
        Args:
            database_url: PostgreSQL connection string
            echo: Enable SQLAlchemy query logging
        """
        self.engine = create_engine(database_url, echo=echo)
        self.SessionLocal = sessionmaker(bind=self.engine)
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)

    def _get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def save_document(self, metadata: DocumentMetadata, raw_text: str, chunks: list[Chunk]) -> UUID:
        """Store document with metadata and chunks."""
        session = self._get_session()
        try:
            # Store document
            doc_record = DocumentRecord(
                id=metadata.id,
                filename=metadata.filename,
                document_type=metadata.document_type.value,
                raw_text=raw_text,
                page_count=metadata.page_count,
            )
            session.add(doc_record)
            session.flush()

            # Store chunks
            for chunk in chunks:
                chunk_record = ChunkRecord(
                    id=chunk.id,
                    document_id=metadata.id,
                    chunk_index=chunk.metadata.paragraph_index or 0,
                    offset=0,
                    length=len(chunk.text),
                    page_num=chunk.metadata.page_number or 0,
                    section_name=chunk.metadata.source_label,
                    text=chunk.text,
                )
                session.add(chunk_record)

            session.commit()
            logger.info(f"Saved document {metadata.id} with {len(chunks)} chunks")
            return metadata.id
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save document: {e}")
            raise
        finally:
            session.close()

    def get_document_by_id(self, document_id: UUID) -> Optional[tuple[DocumentMetadata, list[Chunk]]]:
        """Retrieve document metadata and chunks."""
        session = self._get_session()
        try:
            doc_record = session.query(DocumentRecord).filter(DocumentRecord.id == document_id).first()
            if not doc_record:
                return None

            # Reconstruct metadata
            metadata = DocumentMetadata(
                id=doc_record.id,
                filename=doc_record.filename,
                document_type=doc_record.document_type,
                page_count=doc_record.page_count,
            )

            # Retrieve chunks
            chunk_records = (
                session.query(ChunkRecord).filter(ChunkRecord.document_id == document_id).order_by(ChunkRecord.chunk_index).all()
            )
            chunks = [self._chunk_record_to_chunk(cr) for cr in chunk_records]

            return metadata, chunks
        finally:
            session.close()

    def list_documents(self) -> list[DocumentMetadata]:
        """List all documents."""
        session = self._get_session()
        try:
            doc_records = session.query(DocumentRecord).all()
            return [
                DocumentMetadata(
                    id=dr.id,
                    filename=dr.filename,
                    document_type=dr.document_type,
                    page_count=dr.page_count,
                )
                for dr in doc_records
            ]
        finally:
            session.close()

    def delete_document(self, document_id: UUID) -> None:
        """Delete document and associated chunks."""
        session = self._get_session()
        try:
            session.query(ChunkRecord).filter(ChunkRecord.document_id == document_id).delete()
            session.query(DocumentRecord).filter(DocumentRecord.id == document_id).delete()
            session.commit()
            logger.info(f"Deleted document {document_id}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete document: {e}")
            raise
        finally:
            session.close()

    def get_chunks_by_document(self, document_id: UUID) -> list[Chunk]:
        """Retrieve all chunks for a document."""
        session = self._get_session()
        try:
            chunk_records = (
                session.query(ChunkRecord).filter(ChunkRecord.document_id == document_id).order_by(ChunkRecord.chunk_index).all()
            )
            return [self._chunk_record_to_chunk(cr) for cr in chunk_records]
        finally:
            session.close()

    def save_chunk_embedding(self, chunk_id: UUID, embedding: list[float]) -> None:
        """Store vector embedding for a chunk."""
        session = self._get_session()
        try:
            import json

            chunk_record = session.query(ChunkRecord).filter(ChunkRecord.id == chunk_id).first()
            if chunk_record:
                chunk_record.embedding = json.dumps(embedding)
                session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save embedding: {e}")
            raise
        finally:
            session.close()

    def search_chunks_by_embedding(self, query_embedding: list[float], top_k: int = 5, document_ids: Optional[list[UUID]] = None) -> list[StoredChunk]:
        """
        Retrieve top-k chunks by vector similarity.
        Computes cosine similarity in memory.
        """
        import math
        
        def cosine_similarity(v1: list[float], v2: list[float]) -> float:
            if not v1 or not v2:
                return 0.0
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)

        session = self._get_session()
        try:
            query = session.query(ChunkRecord)
            if document_ids:
                query = query.filter(ChunkRecord.document_id.in_(document_ids))
                
            chunk_records = query.all()
            scored_chunks = []
            
            for cr in chunk_records:
                stored = self._chunk_record_to_stored_chunk(cr)
                if stored.embedding:
                    score = cosine_similarity(query_embedding, stored.embedding)
                    if score > 0.65:  # Strict filter
                        scored_chunks.append((stored, score))
                        
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            return [chunk for chunk, score in scored_chunks][:top_k]
        finally:
            session.close()

    def search_chunks_by_keyword(self, keywords: list[str], document_id: Optional[UUID] = None, top_k: int = 5) -> list[StoredChunk]:
        """Retrieve top-k chunks by keyword matching."""
        session = self._get_session()
        try:
            query = session.query(ChunkRecord)

            if document_id:
                query = query.filter(ChunkRecord.document_id == document_id)

            # Simple keyword matching (full-text search in Phase 5)
            for keyword in keywords:
                query = query.filter(ChunkRecord.text.ilike(f"%{keyword}%"))

            chunk_records = query.limit(top_k).all()
            return [self._chunk_record_to_stored_chunk(cr) for cr in chunk_records]
        finally:
            session.close()

    @staticmethod
    def _chunk_record_to_chunk(record: ChunkRecord) -> Chunk:
        """Convert database record to Chunk domain model."""
        metadata = ChunkMetadata(
            id=record.id,
            document_id=record.document_id,
            page_number=int(record.page_num),
            paragraph_index=int(record.chunk_index),
            source_label=record.section_name or "",
        )
        return Chunk(text=record.text, metadata=metadata)

    @staticmethod
    def _chunk_record_to_stored_chunk(record: ChunkRecord) -> StoredChunk:
        """Convert database record to StoredChunk."""
        import json

        embedding = None
        if record.embedding:
            try:
                embedding = json.loads(record.embedding)
            except (json.JSONDecodeError, TypeError):
                pass

        metadata = ChunkMetadata(
            id=record.id,
            document_id=record.document_id,
            page_number=int(record.page_num),
            paragraph_index=int(record.chunk_index),
            source_label=record.section_name or "",
        )
        return StoredChunk(
            id=record.id,
            document_id=record.document_id,
            text=record.text,
            metadata=metadata,
            embedding=embedding,
        )
