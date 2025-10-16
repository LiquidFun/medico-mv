from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

from embedder import Embedder
from vector_store import VectorStore
from chunker import Chunker
from parsers import DocumentParser
from models import IndexRequest, SearchRequest, SearchResult, DocumentChunk, DocumentInfo

load_dotenv()

# Set Hugging Face cache directory to avoid permission issues
if not os.getenv("HF_HOME"):
    os.environ["HF_HOME"] = "./data/model_cache"

# Initialize FastAPI
app = FastAPI(title="RAG Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (loaded on startup)
embedder: Embedder = None
vector_store: VectorStore = None
chunker: Chunker = None
parser: DocumentParser = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global embedder, vector_store, chunker, parser

    print("=" * 60)
    print("Starting RAG Service...")
    print("=" * 60)

    embedder = Embedder(os.getenv("MODEL_NAME", "sentence-transformers/all-mpnet-base-v2"))
    vector_store = VectorStore(os.getenv("QDRANT_PATH", "./data/qdrant"))
    chunker = Chunker(chunk_size=500, overlap=50)
    parser = DocumentParser()

    print("=" * 60)
    print("✓ RAG Service ready!")
    print("=" * 60)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "rag-service"}


@app.post("/index")
async def index_document(request: IndexRequest):
    """
    Index a document

    - Parses the document
    - Chunks the text
    - Generates embeddings
    - Stores in vector database
    """
    try:
        # Parse document
        text = parser.parse(request.file_path)

        if not text.strip():
            raise HTTPException(status_code=400, detail="Document is empty")

        # Chunk text
        chunks = chunker.chunk_text(
            text,
            metadata={
                "filename": os.path.basename(request.file_path),
                **request.metadata
            }
        )

        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.embed(texts)

        # Store in vector database
        num_chunks = vector_store.add_chunks(request.doc_id, chunks, embeddings)

        return {
            "success": True,
            "doc_id": request.doc_id,
            "num_chunks": num_chunks,
            "message": f"Indexed {num_chunks} chunks"
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResult)
async def search(request: SearchRequest):
    """
    Search for relevant chunks

    - Generates query embedding
    - Searches vector database
    - Returns top-k most similar chunks
    """
    try:
        # Generate query embedding
        query_embedding = embedder.embed_single(request.query)

        # Search vector database
        results = vector_store.search(query_embedding, limit=request.top_k)

        # Format response
        chunks = [
            DocumentChunk(
                doc_id=r["doc_id"],
                chunk_id=r["chunk_id"],
                text=r["text"],
                metadata=r["metadata"]
            )
            for r in results
        ]
        scores = [r["score"] for r in results]

        return SearchResult(chunks=chunks, scores=scores)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its chunks"""
    try:
        vector_store.delete_document(doc_id)
        return {"success": True, "message": f"Deleted document: {doc_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    """List all indexed documents"""
    try:
        docs = vector_store.list_documents()
        return [DocumentInfo(**doc) for doc in docs]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/all")
async def clear_all_documents():
    """Clear all documents from the database"""
    try:
        vector_store.clear_all()
        return {"success": True, "message": "All documents cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8001)),
        reload=True
    )
