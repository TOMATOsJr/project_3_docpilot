"""
Phase 2: Document Adapters

Concrete implementations of the DocumentAdapter ABC for different file formats.
Uses Docling for heavy-lifting (PDF, DOCX) and fallback for lightweight formats.
"""

from __future__ import annotations

import logging
import tempfile
from typing import Any
from pathlib import Path

from app.core.abstractions import DocumentAdapter
from app.core.models import DocumentType

logger = logging.getLogger(__name__)


# ============================================================================
# DOCLING-BASED ADAPTERS (PDF, DOCX)
# ============================================================================


class DoclingSupportedAdapter(DocumentAdapter):
    """Base adapter using Docling for standard document formats."""

    def __init__(self):
        """Initialize with lazy import of Docling to keep optional."""
        try:
            from docling.document_converter import DocumentConverter
            self.converter = DocumentConverter()
            self.docling_available = True
        except ImportError:
            logger.warning("Docling not installed; using fallback text extraction")
            self.converter = None
            self.docling_available = False

    def extract_text(self, filename: str, content: bytes) -> str:
        """Extract plain text from document."""
        if self.docling_available and self.converter:
            try:
                with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as temp_file:
                    temp_file.write(content)
                    temp_path = Path(temp_file.name)
                try:
                    doc = self.converter.convert(str(temp_path))
                    return doc.document.export_to_markdown()
                finally:
                    temp_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Docling extraction failed for {filename}: {e}, using fallback")
                return self._fallback_extract_text(content)
        return self._fallback_extract_text(content)

    def extract_sections(self, filename: str, content: bytes) -> list[dict[str, Any]]:
        """Extract structured sections with page numbers."""
        if self.docling_available and self.converter:
            try:
                with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as temp_file:
                    temp_file.write(content)
                    temp_path = Path(temp_file.name)
                try:
                    doc = self.converter.convert(str(temp_path))
                    sections = []
                    for page_num, page in enumerate(doc.pages, start=1):
                        page_text = page.export_to_markdown()
                        sections.append(
                            {
                                "text": page_text,
                                "page_num": page_num,
                                "section_name": f"Page {page_num}",
                            }
                        )
                    return sections if sections else [{"text": self.extract_text(filename, content), "page_num": 1, "section_name": "Full Document"}]
                finally:
                    temp_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Docling section extraction failed: {e}")
                return self._fallback_extract_sections(content)
        return self._fallback_extract_sections(content)

    def _fallback_extract_text(self, content: bytes) -> str:
        """Fallback: extract text as-is for binary documents."""
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return "[Binary content - manual inspection required]"

    def _fallback_extract_sections(self, content: bytes) -> list[dict[str, Any]]:
        """Fallback: return single section."""
        return [{
            "text": self._fallback_extract_text(content),
            "page_num": 1,
            "section_name": "Full Document",
        }]


class PdfAdapter(DoclingSupportedAdapter):
    """Adapter for PDF documents using Docling."""

    supported_types = [DocumentType.pdf]


class DocxAdapter(DoclingSupportedAdapter):
    """Adapter for DOCX documents using Docling."""

    supported_types = [DocumentType.docx]


class PowerPointAdapter(DoclingSupportedAdapter):
    """Adapter for PowerPoint presentations using Docling."""

    supported_types = [DocumentType.pptx]


# ============================================================================
# LIGHTWEIGHT ADAPTERS (Markdown, Plain Text)
# ============================================================================


class MarkdownAdapter(DocumentAdapter):
    """Lightweight adapter for Markdown documents."""

    supported_types = [DocumentType.markdown]

    def extract_text(self, filename: str, content: bytes) -> str:
        """Decode markdown as UTF-8."""
        return content.decode("utf-8", errors="replace")

    def extract_sections(self, filename: str, content: bytes) -> list[dict[str, Any]]:
        """Split markdown by heading levels."""
        text = self.extract_text(filename, content)
        sections = []
        current_section = ""
        section_name = "Preamble"

        for line in text.split("\n"):
            if line.startswith("# "):
                if current_section:
                    sections.append({
                        "text": current_section.strip(),
                        "page_num": len(sections) + 1,
                        "section_name": section_name,
                    })
                section_name = line[2:].strip()
                current_section = ""
            else:
                current_section += line + "\n"

        if current_section:
            sections.append({
                "text": current_section.strip(),
                "page_num": len(sections) + 1,
                "section_name": section_name,
            })

        return sections if sections else [{
            "text": text,
            "page_num": 1,
            "section_name": filename,
        }]


class PlainTextAdapter(DocumentAdapter):
    """Lightweight adapter for plain text files."""

    supported_types = [DocumentType.text]

    def extract_text(self, filename: str, content: bytes) -> str:
        """Decode plain text as UTF-8."""
        return content.decode("utf-8", errors="replace")

    def extract_sections(self, filename: str, content: bytes) -> list[dict[str, Any]]:
        """Return entire text as one section."""
        return [{
            "text": self.extract_text(filename, content),
            "page_num": 1,
            "section_name": filename,
        }]


# ============================================================================
# ADAPTER REGISTRY & FACTORY
# ============================================================================


class AdapterRegistry:
    """
    Factory for retrieving appropriate adapter by file type.
    Enables swappable, pluggable format support.
    """

    _adapter_factories: dict[DocumentType, type[DocumentAdapter]] = {
        DocumentType.pdf: PdfAdapter,
        DocumentType.docx: DocxAdapter,
        DocumentType.pptx: PowerPointAdapter,
        DocumentType.markdown: MarkdownAdapter,
        DocumentType.text: PlainTextAdapter,
    }
    _adapters: dict[DocumentType, DocumentAdapter] = {}

    @classmethod
    def get_adapter(cls, document_type: DocumentType) -> DocumentAdapter:
        """Get adapter for document type, fallback to plain text."""
        if document_type in cls._adapters:
            return cls._adapters[document_type]

        factory = cls._adapter_factories.get(document_type)
        if factory is None:
            return PlainTextAdapter()

        adapter = factory()
        cls._adapters[document_type] = adapter
        return adapter

    @classmethod
    def get_adapter_for_file(cls, filename: str) -> DocumentAdapter:
        """Get adapter by filename suffix."""
        suffix = Path(filename).suffix.lower()
        doc_type = DocumentAdapter._filename_to_doctype(suffix)
        return cls.get_adapter(doc_type)

    @classmethod
    def register_adapter(cls, document_type: DocumentType, adapter: DocumentAdapter) -> None:
        """Register custom adapter for a document type."""
        cls._adapters[document_type] = adapter
