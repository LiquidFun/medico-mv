import httpx
import os


class RAGClient:
    """HTTP client for RAG service"""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("RAG_SERVICE_URL", "http://indus:8123")

    async def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search for relevant context chunks

        Returns: List of chunk dictionaries with text and metadata
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    json={"query": query, "top_k": top_k}
                )
                response.raise_for_status()
                result = response.json()
                return result.get("chunks", [])
        except Exception as e:
            print(f"RAG search error: {e}")
            return []

    async def search_with_metadata(self, query: str, top_k: int = 3) -> dict:
        """
        Search and return chunks with citation source mapping

        Returns:
            {
                "chunks": [...],  # Original chunks
                "sources": {      # Mapping for citations
                    "1": {
                        "doc_id": "...",
                        "filename": "...",
                        "page": 5,
                        "chunk_id": 0
                    },
                    ...
                }
            }
        """
        chunks = await self.search(query, top_k)

        # Build source mapping
        sources = {}
        for idx, chunk in enumerate(chunks, start=1):
            sources[str(idx)] = {
                "doc_id": chunk.get("doc_id", "unknown"),
                "filename": chunk.get("metadata", {}).get("filename", "Unknown"),
                "page": chunk.get("metadata", {}).get("page", 1),
                "chunk_id": chunk.get("chunk_id", 0),
                "text": chunk.get("text", "")  # Include the chunk text
            }

        return {"chunks": chunks, "sources": sources}

    async def get_document_file(self, doc_id: str) -> bytes:
        """Fetch PDF file from RAG service"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.base_url}/document/{doc_id}/file")
                response.raise_for_status()
                return response.content
        except Exception as e:
            print(f"RAG document fetch error: {e}")
            raise

    async def health_check(self) -> bool:
        """Check if RAG service is available"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except:
            return False
