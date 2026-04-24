"""
Phase 2: Document Ingestion Service

High-level service that orchestrates document parsing, chunking, and storage.
Uses the Adapter and Strategy patterns to make ingestion pluggable.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from app.core.abstractions import DocumentAdapter, ChunkingStrategy, DocumentRepository
from app.core.models import Chunk, DocumentMetadata
from app.core.adapters import AdapterRegistry
from app.core.strategies import ChunkingStrategyRegistry

logger = logging.getLogger(__name__)


class IngestService:
    """
    Orchestrates the complete document ingestion pipeline:
    1. Detect format and get appropriate adapter
    2. Parse document to extract text
    3. Chunk text using selected strategy
    4. Store in repository
    """

    def __init__(
        self,
        repository: DocumentRepository,
        adapter_registry: Optional[AdapterRegistry] = None,
        chunking_registry: Optional[ChunkingStrategyRegistry] = None,
    ):
        """
        Initialize ingestion service.
        Args:
            repository: DocumentRepository implementation for persistence
            adapter_registry: Custom adapter registry (uses default if None)
            chunking_registry: Custom chunking registry (uses default if None)
        """
        self.repository = repository
        self.adapter_registry = adapter_registry or AdapterRegistry
        self.chunking_registry = chunking_registry or ChunkingStrategyRegistry

    def ingest(
        self,
        filename: str,
        content: bytes,
        chunking_strategy: str = "paragraph",
    ) -> UUID:
        """
        Ingest a document file end-to-end.
        Args:
            filename: Original filename (used to detect format)
            content: Raw file bytes
            chunking_strategy: Name of chunking strategy to use
        Returns:
            document_id of stored document
        """
        logger.info(f"Ingesting {filename} with strategy '{chunking_strategy}'")

        # Step 1: Get adapter
        adapter = self.adapter_registry.get_adapter_for_file(filename)
        logger.debug(f"Using adapter: {adapter.__class__.__name__}")

        # Step 2: Extract text
        text = adapter.extract_text(filename, content)
        logger.debug(f"Extracted {len(text)} characters")

        # Step 3: Create metadata
        from app.core.models import DocumentType
        from pathlib import Path

        suffix = Path(filename).suffix.lower()
        doc_type = DocumentAdapter._filename_to_doctype(suffix)
        metadata = DocumentMetadata(
            filename=filename,
            document_type=doc_type,
            page_count=len(adapter.extract_sections(filename, content)),
        )

        # Step 4: Chunk text
        strategy = self.chunking_registry.get_strategy(chunking_strategy)
        chunks = strategy.chunk(metadata.id, text)
        logger.debug(f"Created {len(chunks)} chunks")

        # Step 5: Store
        doc_id = self.repository.save_document(metadata, text, chunks)
        logger.info(f"Stored document {doc_id} with {len(chunks)} chunks")

        # Step 6: Generate and save embeddings
        self._embed_and_save_chunks(chunks)

        return doc_id

    def ingest_from_sections(
        self,
        filename: str,
        content: bytes,
        chunking_strategy: str = "paragraph",
    ) -> UUID:
        """
        Ingest using structured sections extracted by adapter.
        Useful for documents with natural page/section boundaries.
        """
        logger.info(f"Ingesting {filename} with sections and strategy '{chunking_strategy}'")

        # Get adapter
        adapter = self.adapter_registry.get_adapter_for_file(filename)

        # Extract sections
        sections = adapter.extract_sections(filename, content)
        logger.debug(f"Extracted {len(sections)} sections")

        # Create metadata
        from app.core.models import DocumentType
        from pathlib import Path

        suffix = Path(filename).suffix.lower()
        doc_type = DocumentAdapter._filename_to_doctype(suffix)
        metadata = DocumentMetadata(
            filename=filename,
            document_type=doc_type,
            page_count=len(sections),
        )

        # Chunk each section separately
        strategy = self.chunking_registry.get_strategy(chunking_strategy)
        all_chunks: list[Chunk] = []

        for section in sections:
            section_chunks = strategy.chunk(
                metadata.id,
                section["text"],
                page_num=section.get("page_num", 1),
            )
            # Preserve section name in metadata
            for chunk in section_chunks:
                chunk.metadata.source_label = section.get("section_name", "")
            all_chunks.extend(section_chunks)

        # Re-index chunks
        for i, chunk in enumerate(all_chunks):
            chunk.metadata.paragraph_index = i

        # Store
        doc_id = self.repository.save_document(metadata, "\n".join(s["text"] for s in sections), all_chunks)
        logger.info(f"Stored document {doc_id} with {len(all_chunks)} chunks from {len(sections)} sections")

        # Generate and save embeddings
        self._embed_and_save_chunks(all_chunks)

        return doc_id

    def _embed_and_save_chunks(self, chunks: list[Chunk]) -> None:
        """Helper to generate and save embeddings for a list of chunks."""
        if not chunks:
            return
            
        from app.config import get_settings
        import litellm
        
        settings = get_settings()
        
        try:
            # Generate embeddings for all chunks in a single batch (or multiple if large, but litellm handles reasonable lists)
            inputs = [chunk.text for chunk in chunks]
            response = litellm.embedding(model=settings.embedding_model, input=inputs)
            
            for i, chunk in enumerate(chunks):
                if i < len(response.data):
                    embedding = response.data[i]['embedding']
                    self.repository.save_chunk_embedding(chunk.id, embedding)
            logger.info(f"Successfully generated embeddings for {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
