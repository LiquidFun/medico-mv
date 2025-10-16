from sentence_transformers import SentenceTransformer
import numpy as np
import os


class Embedder:
    """Wrapper for sentence-transformers model with GPU support"""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        print(f"Loading embedding model: {model_name}...")

        # Get GPU device from environment variable (default: cuda:0)
        device = os.getenv("CUDA_DEVICE", "cuda:0")

        self.model = SentenceTransformer(model_name, device=device)
        print(f"✓ Embedding model loaded on {device}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        return self.embed([text])[0]
