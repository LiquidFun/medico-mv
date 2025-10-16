from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import shutil
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
    Index a document by file path (for local files on the server)

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


@app.post("/upload")
async def upload_and_index(
    file: UploadFile = File(...),
    doc_id: str = Form(None),
    metadata: str = Form("{}")
):
    """
    Upload and index a document

    - Receives file upload
    - Saves to temporary location
    - Parses, chunks, embeds, and indexes
    - Cleans up temporary file
    """
    import json

    # Use filename (without extension) as doc_id if not provided
    if doc_id is None:
        doc_id = os.path.splitext(file.filename)[0]

    # Parse metadata JSON
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError:
        metadata_dict = {}

    # Add filename to metadata
    metadata_dict["filename"] = file.filename

    # Create temporary file
    temp_file = None
    try:
        # Create temp file with same extension as uploaded file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            # Copy uploaded file to temp location
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # Parse document
        text = parser.parse(temp_path)

        if not text.strip():
            raise HTTPException(status_code=400, detail="Document is empty")

        # Chunk text
        chunks = chunker.chunk_text(text, metadata=metadata_dict)

        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.embed(texts)

        # Store in vector database
        num_chunks = vector_store.add_chunks(doc_id, chunks, embeddings)

        return {
            "success": True,
            "doc_id": doc_id,
            "num_chunks": num_chunks,
            "message": f"Indexed {num_chunks} chunks from {file.filename}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_path):
            os.unlink(temp_path)


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
