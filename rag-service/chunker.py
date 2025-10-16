import re


class Chunker:
    """Text chunking with overlap"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, metadata: dict) -> list[dict]:
        """
        Split text into overlapping chunks with metadata

        Returns: [{"text": "...", "metadata": {...}}, ...]
        """
        # Simple word-based chunking
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "metadata": {**metadata, "chunk_index": len(chunks)}
                })

        return chunks if chunks else [{"text": text, "metadata": {**metadata, "chunk_index": 0}}]
