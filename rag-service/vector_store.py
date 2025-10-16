from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
from datetime import datetime
import uuid


class VectorStore:
    """Qdrant vector database client"""

    def __init__(self, path: str = "./data/qdrant"):
        print(f"Initializing Qdrant at {path}...")
        self.client = QdrantClient(path=path)
        self.collection_name = "documents"
        self._init_collection()
        print("✓ Qdrant initialized")

    def _init_collection(self):
        """Initialize collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            print(f"✓ Created collection: {self.collection_name}")

    def add_chunks(self, doc_id: str, chunks: list[dict], embeddings: np.ndarray):
        """Add document chunks with embeddings"""
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "doc_id": doc_id,
                        "chunk_id": i,
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                        "indexed_at": datetime.utcnow().isoformat(),
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search(self, query_embedding: np.ndarray, limit: int = 5) -> list[dict]:
        """Search for similar chunks"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
        )

        return [
            {
                "doc_id": hit.payload["doc_id"],
                "chunk_id": hit.payload["chunk_id"],
                "text": hit.payload["text"],
                "metadata": hit.payload["metadata"],
                "score": hit.score,
            }
            for hit in results
        ]

    def delete_document(self, doc_id: str):
        """Delete all chunks from a document"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector={"filter": {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}},
        )

    def list_documents(self) -> list[dict]:
        """List all unique documents"""
        # Scroll through all points and collect unique doc_ids
        docs = {}
        offset = None

        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for record in records:
                doc_id = record.payload["doc_id"]
                if doc_id not in docs:
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "filename": record.payload["metadata"].get("filename", doc_id),
                        "num_chunks": 0,
                        "uploaded_at": record.payload.get("indexed_at", ""),
                    }
                docs[doc_id]["num_chunks"] += 1

            if offset is None:
                break

        return list(docs.values())

    def clear_all(self):
        """Delete all documents"""
        self.client.delete_collection(self.collection_name)
        self._init_collection()
