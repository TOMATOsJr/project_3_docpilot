"""
Phase 2: Chunking Strategies

Concrete implementations of ChunkingStrategy for different text segmentation policies.
Enables swappable chunking algorithms for different document types and use cases.
"""

from __future__ import annotations

import re
from uuid import UUID

from app.core.abstractions import ChunkingStrategy
from app.core.models import Chunk


class ParagraphChunkingStrategy(ChunkingStrategy):
    """
    Strategy: Split text into paragraphs (natural semantic boundaries).
    Preserves paragraph structure; good for documents with clear section breaks.
    """

    def __init__(self, max_chunk_size: int = 1024):
        """
        Args:
            max_chunk_size: Max characters per chunk. If a paragraph exceeds this, split further.
        """
        self.max_chunk_size = max_chunk_size

    def chunk(self, document_id: UUID, text: str, page_num: int = 1) -> list[Chunk]:
        """Split text into paragraph-based chunks."""
        # Split by double newline or multiple spaces (paragraph breaks)
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        chunk_index = 0
        offset = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If paragraph is too large, split further by sentences
            if len(para) > self.max_chunk_size:
                for sentence_chunk in self._split_sentences(para, self.max_chunk_size):
                    chunk = self._create_chunk(
                        document_id=document_id,
                        text=sentence_chunk.strip(),
                        offset=offset,
                        length=len(sentence_chunk),
                        page_num=page_num,
                        section=f"Paragraph {chunk_index}",
                    )
                    chunks.append(chunk)
                    offset += len(sentence_chunk)
                    chunk_index += 1
            else:
                chunk = self._create_chunk(
                    document_id=document_id,
                    text=para,
                    offset=offset,
                    length=len(para),
                    page_num=page_num,
                    section=f"Paragraph {chunk_index}",
                )
                chunks.append(chunk)
                offset += len(para)
                chunk_index += 1

        # Update paragraph indices
        for i, chunk in enumerate(chunks):
            chunk.metadata.paragraph_index = i

        return chunks

    @staticmethod
    def _split_sentences(text: str, max_size: int) -> list[str]:
        """Split text into sentences, respecting max_size limit."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = ""

        for sent in sentences:
            if len(current_chunk) + len(sent) <= max_size:
                current_chunk += " " + sent if current_chunk else sent
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sent

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


class TokenChunkingStrategy(ChunkingStrategy):
    """
    Strategy: Split text into fixed-size token chunks.
    Approximates tokens by word count (1 token ≈ 4 chars).
    Good for ensuring consistent chunk size for embedding models.
    """

    def __init__(self, chunk_size_tokens: int = 256, overlap_tokens: int = 32):
        """
        Args:
            chunk_size_tokens: Target chunk size in approximate tokens.
            overlap_tokens: Overlap between adjacent chunks to preserve context.
        """
        self.chunk_size_chars = chunk_size_tokens * 4  # Approximate
        self.overlap_chars = overlap_tokens * 4

    def chunk(self, document_id: UUID, text: str, page_num: int = 1) -> list[Chunk]:
        """Split text into fixed-size overlapping chunks."""
        chunks = []
        chunk_index = 0
        offset = 0

        while offset < len(text):
            # Extract chunk
            chunk_end = min(offset + self.chunk_size_chars, len(text))
            chunk_text = text[offset : chunk_end].strip()

            if not chunk_text:
                break

            chunk = self._create_chunk(
                document_id=document_id,
                text=chunk_text,
                offset=offset,
                length=len(chunk_text),
                page_num=page_num,
                section=f"Chunk {chunk_index}",
            )
            chunk.metadata.paragraph_index = chunk_index
            chunks.append(chunk)

            # Move offset with overlap
            offset = chunk_end - self.overlap_chars if chunk_end < len(text) else len(text)
            chunk_index += 1

        return chunks


class SlidingWindowChunkingStrategy(ChunkingStrategy):
    """
    Strategy: Sliding window with configurable step and window size.
    Maximal overlap; useful for retrieval-heavy workloads.
    """

    def __init__(self, window_size: int = 512, step_size: int = 128):
        """
        Args:
            window_size: Characters per window.
            step_size: Stride for window movement.
        """
        self.window_size = window_size
        self.step_size = step_size

    def chunk(self, document_id: UUID, text: str, page_num: int = 1) -> list[Chunk]:
        """Create sliding window chunks."""
        chunks = []
        chunk_index = 0

        for offset in range(0, len(text), self.step_size):
            chunk_end = min(offset + self.window_size, len(text))
            chunk_text = text[offset : chunk_end].strip()

            if not chunk_text:
                continue

            chunk = self._create_chunk(
                document_id=document_id,
                text=chunk_text,
                offset=offset,
                length=len(chunk_text),
                page_num=page_num,
                section=f"Window {chunk_index}",
            )
            chunk.metadata.paragraph_index = chunk_index
            chunks.append(chunk)
            chunk_index += 1

        return chunks


# ============================================================================
# STRATEGY REGISTRY & FACTORY
# ============================================================================


class ChunkingStrategyRegistry:
    """Factory for retrieving chunking strategy by name."""

    _strategies: dict[str, ChunkingStrategy] = {
        "paragraph": ParagraphChunkingStrategy(),
        "token": TokenChunkingStrategy(),
        "sliding_window": SlidingWindowChunkingStrategy(),
    }

    @classmethod
    def get_strategy(cls, name: str) -> ChunkingStrategy:
        """Get strategy by name, default to paragraph."""
        return cls._strategies.get(name, ParagraphChunkingStrategy())

    @classmethod
    def register_strategy(cls, name: str, strategy: ChunkingStrategy) -> None:
        """Register custom chunking strategy."""
        cls._strategies[name] = strategy
