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

    async def health_check(self) -> bool:
        """Check if RAG service is available"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except:
            return False
