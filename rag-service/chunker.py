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

    def chunk_pages(self, pages: list[dict], base_metadata: dict) -> list[dict]:
        """
        Chunk pages while preserving page numbers

        Args:
            pages: List of {"page_number": int, "text": str}
            base_metadata: Base metadata to include in all chunks

        Returns: List of chunks with page metadata
        """
        all_chunks = []

        for page_data in pages:
            page_num = page_data["page_number"]
            text = page_data["text"]

            # Create metadata with page number
            page_metadata = {
                **base_metadata,
                "page": page_num
            }

            # Chunk this page's text
            page_chunks = self.chunk_text(text, page_metadata)
            all_chunks.extend(page_chunks)

        return all_chunks
