"""
Chunker Module: Overlapping Sliding-Window Text Chunking
--------------------------------------------------------
Implements fixed-size sliding-window chunking with token/word overlap.
Ensures semantic continuity across chunk boundaries in the RAG pipeline.
"""

from typing import List, Dict, Any
import re


class TextChunker:
    def __init__(self, chunk_size: int = 180, chunk_overlap: int = 40):
        """
        Args:
            chunk_size (int): Target number of words per chunk.
            chunk_overlap (int): Number of overlapping words between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Splits text into overlapping chunks and attaches metadata.
        """
        if not text:
            return []

        # Normalize whitespace
        clean_text = re.sub(r"\s+", " ", text).strip()
        words = clean_text.split()
        if not words:
            return []

        chunks = []
        start = 0
        chunk_idx = 0
        base_meta = metadata or {}

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            # Skip tiny tail fragments
            if len(chunk_words) >= 20 or len(chunks) == 0:
                chunk_entry = {
                    "chunk_id": chunk_idx,
                    "text": chunk_text,
                    "word_count": len(chunk_words),
                    "source_url": base_meta.get("url", ""),
                    "source_title": base_meta.get("title", ""),
                    "source_type": base_meta.get("source_type", "scraped"),
                }
                chunks.append(chunk_entry)
                chunk_idx += 1

            if end == len(words):
                break
            start += max(1, (self.chunk_size - self.chunk_overlap))

        return chunks
