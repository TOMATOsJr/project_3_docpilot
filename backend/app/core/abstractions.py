"""
Phase 0: Architecture Contracts

Defines abstract base classes for core components, establishing interfaces
that enable pluggable implementations throughout DocPilot. These contracts
ensure dependency inversion and enable the pattern-driven architecture.

Patterns applied:
- Adapter: DocumentAdapter for format-specific parsing
- Strategy: ChunkingStrategy and ModelProvider for swappable algorithms
- Chain of Responsibility: ModelProvider.fallback_chain for model redundancy
- Facade: ModelGateway (concrete) uses ModelProvider contracts
- Command + Memento: EditCommand and StateSnapshot for reversibility
- Repository: DocumentRepository for storage abstraction
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from app.core.models import (
    Chunk,
    ChunkMetadata,
    Citation,
    DocumentMetadata,
    DocumentType,
    EditProposal,
    EditRequest,
    QueryResponse,
)


# ============================================================================
# ADAPTER PATTERN: Document Parsing
# ============================================================================


class DocumentAdapter(ABC):
    """
    Abstract adapter for format-specific document parsing.
    Concrete implementations: PdfAdapter, DocxAdapter, PowerPointAdapter, MarkdownAdapter.
    Responsibility: Read raw bytes, extract structured text with page/section metadata.
    """

    supported_types: list[DocumentType]

    @abstractmethod
    def extract_text(self, filename: str, content: bytes) -> str:
        """
        Extract plain text from document bytes.
        Should preserve logical structure (paragraphs, lists, headings).
        """
        pass

    @abstractmethod
    def extract_sections(self, filename: str, content: bytes) -> list[dict[str, Any]]:
        """
        Extract structured sections/pages with metadata.
        Returns list of dicts with keys: 'text', 'page_num', 'section_name', etc.
        """
        pass

    def can_parse(self, filename: str) -> bool:
        """Check if this adapter can parse the given filename."""
        from pathlib import Path

        suffix = Path(filename).suffix.lower()
        doc_type = self._filename_to_doctype(suffix)
        return doc_type in self.supported_types

    @staticmethod
    def _filename_to_doctype(suffix: str) -> DocumentType:
        """Map file suffix to DocumentType enum."""
        suffix_map = {
            ".pdf": DocumentType.pdf,
            ".docx": DocumentType.docx,
            ".pptx": DocumentType.pptx,
            ".md": DocumentType.markdown,
            ".txt": DocumentType.text,
        }
        return suffix_map.get(suffix, DocumentType.unknown)


# ============================================================================
# STRATEGY PATTERN: Chunking
# ============================================================================


class ChunkingStrategy(ABC):
    """
    Abstract strategy for splitting text into chunks.
    Concrete implementations: ParagraphChunkingStrategy, TokenChunkingStrategy, SlidingWindowChunkingStrategy.
    Responsibility: Apply chunking policy and generate provenance metadata.
    """

    @abstractmethod
    def chunk(self, document_id: UUID, text: str, page_num: int = 1) -> list[Chunk]:
        """
        Split text into chunks according to strategy rules.
        Each chunk should have ChunkMetadata with offset/length for provenance.
        """
        pass

    def _create_chunk(
        self,
        document_id: UUID,
        text: str,
        offset: int,
        length: int,
        page_num: int = 1,
        section: str = "",
    ) -> Chunk:
        """Helper to create a Chunk with full metadata."""
        metadata = ChunkMetadata(
            document_id=document_id,
            page_number=page_num,
            paragraph_index=0,  # Will be updated by caller if needed
            source_label=section,
        )
        return Chunk(text=text, metadata=metadata)


# ============================================================================
# REPOSITORY PATTERN: Data Persistence
# ============================================================================


@dataclass(slots=True)
class StoredChunk:
    """Value object for persisted chunk with all metadata."""

    id: UUID
    document_id: UUID
    text: str
    metadata: ChunkMetadata
    embedding: Optional[list[float]] = None


class DocumentRepository(ABC):
    """
    Abstract repository for document persistence.
    Concrete implementations: InMemoryRepository, PostgresRepository.
    Responsibility: CRUD operations for documents, chunks, and citations.
    Separates storage concerns from business logic (Repository pattern).
    """

    @abstractmethod
    def save_document(self, metadata: DocumentMetadata, raw_text: str, chunks: list[Chunk]) -> UUID:
        """Store a document with its chunks. Returns document ID."""
        pass

    @abstractmethod
    def get_document_by_id(self, document_id: UUID) -> Optional[tuple[DocumentMetadata, list[Chunk]]]:
        """Retrieve document metadata and chunks by ID."""
        pass

    @abstractmethod
    def list_documents(self) -> list[DocumentMetadata]:
        """List all document metadata."""
        pass

    @abstractmethod
    def delete_document(self, document_id: UUID) -> None:
        """Delete document and all associated chunks."""
        pass

    @abstractmethod
    def get_chunks_by_document(self, document_id: UUID) -> list[Chunk]:
        """Retrieve all chunks for a document."""
        pass

    @abstractmethod
    def save_chunk_embedding(self, chunk_id: UUID, embedding: list[float]) -> None:
        """Store vector embedding for a chunk (for Phase 3 semantic search)."""
        pass

    @abstractmethod
    def search_chunks_by_embedding(self, query_embedding: list[float], top_k: int = 5) -> list[StoredChunk]:
        """Retrieve top-k chunks by vector similarity."""
        pass

    @abstractmethod
    def search_chunks_by_keyword(self, keywords: list[str], document_id: Optional[UUID] = None, top_k: int = 5) -> list[StoredChunk]:
        """Retrieve top-k chunks by keyword matching."""
        pass


# ============================================================================
# STRATEGY PATTERN: Context Engine (RAG Pipeline)
# ============================================================================


class ContextEngine(ABC):
    """
    Abstract Context Engine orchestrating the RAG pipeline.
    Responsibility: Select documents, retrieve chunks, compose answers with citations.
    Implements the RAG flow: Retrieve → Rank → Compose → Cite.
    """

    @abstractmethod
    def retrieve_context(
        self,
        query: str,
        document_ids: Optional[list[UUID]] = None,
        top_k: int = 5,
    ) -> list[Chunk]:
        """
        Retrieve relevant chunks matching the query.
        May use keyword, semantic, or hybrid retrieval.
        """
        pass

    @abstractmethod
    def compose_answer(self, query: str, context_chunks: list[Chunk], model_completion: str) -> QueryResponse:
        """
        Compose a final answer with citations.
        Maps model output back to source chunks and builds Citation objects.
        """
        pass


# ============================================================================
# STRATEGY + CHAIN OF RESPONSIBILITY: Model Providers
# ============================================================================


class ModelProvider(ABC):
    """
    Abstract provider for LLM clients.
    Concrete implementations: OpenAIProvider, AnthropicProvider, GroqProvider, LocalProvider.
    Enables Chain of Responsibility via fallback_chain property.
    Responsibility: Wrap vendor-specific client calls, provide uniform interface.
    """

    provider_name: str
    latency_budget_ms: float = 3000.0

    @abstractmethod
    def complete(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """
        Synchronous LLM completion call.
        May raise TimeoutError if latency exceeds budget.
        """
        pass

    @abstractmethod
    async def complete_async(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """
        Asynchronous LLM completion call for non-blocking orchestration.
        Used by agent orchestrator for concurrent requests.
        """
        pass

    def fallback_chain(self) -> list[ModelProvider]:
        """
        Return list of fallback providers in priority order.
        Used by Facade (ModelGateway) for Chain of Responsibility.
        Default: empty list (no fallback).
        """
        return []

    def is_available(self) -> bool:
        """Health check: is this provider reachable and ready?"""
        return True


# ============================================================================
# COMMAND + MEMENTO PATTERN: Reversible Edits
# ============================================================================


@dataclass(slots=True)
class StateSnapshot:
    """
    Memento: captures document state at a point in time.
    Enables undo/redo by restoring to a prior snapshot.
    """

    snapshot_id: UUID
    document_id: UUID
    text_before: str
    text_after: str
    edit_description: str
    timestamp: str  # ISO 8601


class EditCommand(ABC):
    """
    Abstract command for reversible document edits.
    Concrete implementations: ReplaceCommand, InsertCommand, DeleteCommand.
    Implements Command pattern with Memento for undo/redo.
    Responsibility: Execute edit, generate proposal, support undo/redo.
    """

    @abstractmethod
    def execute(self, original_text: str) -> str:
        """Apply the edit and return modified text."""
        pass

    @abstractmethod
    def undo(self, modified_text: str) -> str:
        """Reverse the edit."""
        pass

    @abstractmethod
    def to_proposal(self, original_text: str) -> EditProposal:
        """Generate a human-readable diff proposal."""
        pass

    def create_memento(self, document_id: UUID, original_text: str) -> StateSnapshot:
        """Create a memento capturing the before/after state."""
        from datetime import datetime
        from uuid import uuid4

        modified_text = self.execute(original_text)
        return StateSnapshot(
            snapshot_id=uuid4(),
            document_id=document_id,
            text_before=original_text,
            text_after=modified_text,
            edit_description=str(self),
            timestamp=datetime.utcnow().isoformat(),
        )


# ============================================================================
# STRATEGY PATTERN: Agent Modes
# ============================================================================


class AgentStrategy(ABC):
    """
    Abstract agent strategy for different task modes.
    Concrete implementations: QaAgent, EditAgent, SynthesisAgent, MultiDocAgent.
    Enables Strategy pattern for task-specific orchestration.
    Responsibility: Implement task logic using context engine, model provider, and repository.
    """

    agent_name: str

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the agent's primary task.
        Signature varies by agent (e.g., QaAgent takes query and document_ids).
        Returns task-specific result dict.
        """
        pass

    @abstractmethod
    def validate_input(self, **kwargs: Any) -> bool:
        """Check if inputs are valid before execution."""
        pass


# ============================================================================
# FACADE: Model Gateway (uses ModelProvider contracts)
# ============================================================================


class ModelGateway(ABC):
    """
    Facade providing unified access to multiple model providers.
    Uses ModelProvider contracts via dependency injection.
    Implements Chain of Responsibility via provider fallback_chain.
    Responsibility: Route by task type, handle provider fallback, enforce budgets.
    """

    @abstractmethod
    def complete_for_qa(self, query: str, context: str, temperature: float = 0.5) -> str:
        """Route Q&A requests with optimal model (faster, less expensive)."""
        pass

    @abstractmethod
    def complete_for_synthesis(self, query: str, contexts: list[str], temperature: float = 0.7) -> str:
        """Route synthesis requests with appropriate model (handles longer context)."""
        pass

    @abstractmethod
    def complete_for_edit_proposal(self, edit_instructions: str, text: str, temperature: float = 0.3) -> str:
        """Route edit proposals with precise model (low temp for determinism)."""
        pass

    @abstractmethod
    def with_fallback(self, primary: ModelProvider, fallback_chain: list[ModelProvider]) -> ModelGateway:
        """Configure fallback chain for availability (redundancy pattern)."""
        pass


# ============================================================================
# OBSERVER PATTERN (stub for future)
# ============================================================================


class DocumentChangeListener(ABC):
    """
    Observer interface for document mutations.
    Enables UI to refresh when documents or citations change.
    Concrete implementations: UIUpdateListener, AuditListener.
    """

    @abstractmethod
    def on_document_uploaded(self, document_id: UUID) -> None:
        """Fired when a new document is stored."""
        pass

    @abstractmethod
    def on_document_deleted(self, document_id: UUID) -> None:
        """Fired when a document is removed."""
        pass

    @abstractmethod
    def on_citation_added(self, document_id: UUID, chunk_id: UUID) -> None:
        """Fired when a chunk is cited in an answer."""
        pass
