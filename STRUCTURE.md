# RAG System Implementation Structure

## Overview
Two-tier architecture: GPU server for embeddings/search, chat app calls it via HTTP.

## Project Structure

```
/workspace/chat/
├── ai-chat-app/              # Existing chat app
│   ├── app/
│   │   ├── services/
│   │   │   └── rag_client.py      # NEW: HTTP client to GPU server
│   │   └── routes/
│   │       └── documents.py       # NEW: Document upload API
│   └── templates/
│       └── index.html             # MODIFY: Add upload UI
│
└── rag-service/              # NEW: GPU server service
    ├── pyproject.toml        # uv managed
    ├── main.py               # FastAPI app
    ├── embedder.py           # Embedding model wrapper
    ├── vector_store.py       # Qdrant client
    ├── chunker.py            # Text chunking
    ├── parsers.py            # PDF/txt/docx parsing
    ├── models.py             # Pydantic models
    ├── cli.py                # Document ingestion CLI
    └── data/                 # Document storage
        ├── uploads/          # Original files
        └── qdrant/           # Vector DB data
```

---

## RAG Service (GPU Server)

### Dependencies (pyproject.toml)
```toml
[project]
name = "rag-service"
dependencies = [
    "fastapi>=0.110.0",
    "uvicorn>=0.27.0",
    "sentence-transformers>=2.5.0",
    "qdrant-client>=1.8.0",
    "pymupdf>=1.23.0",        # PDF parsing
    "python-docx>=1.1.0",     # DOCX parsing
    "click>=8.1.0",           # CLI
    "python-dotenv>=1.0.0",
    "torch>=2.0.0",
]
```

### API Endpoints

```python
# main.py - FastAPI app

POST   /index              # Index a document (from CLI or chat app)
POST   /search             # Search for relevant chunks
DELETE /document/{doc_id}  # Delete document
GET    /documents          # List all documents
GET    /health             # Health check
```

### Core Classes

#### 1. Embedder (embedder.py)
```python
class Embedder:
    """Wrapper for sentence-transformers model"""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name, device="cuda")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        return self.model.encode(texts, convert_to_numpy=True)
```

#### 2. VectorStore (vector_store.py)
```python
class VectorStore:
    """Qdrant vector database client"""

    def __init__(self, path: str = "./data/qdrant"):
        self.client = QdrantClient(path=path)
        self._init_collection()

    def add_chunks(self, doc_id: str, chunks: list[dict], embeddings: np.ndarray):
        """Add document chunks with embeddings"""

    def search(self, query_embedding: np.ndarray, limit: int = 5) -> list[dict]:
        """Search for similar chunks"""

    def delete_document(self, doc_id: str):
        """Delete all chunks from a document"""
```

#### 3. Chunker (chunker.py)
```python
class Chunker:
    """Text chunking with overlap"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, metadata: dict) -> list[dict]:
        """Split text into overlapping chunks with metadata"""
        # Returns: [{"text": "...", "metadata": {...}}, ...]
```

#### 4. DocumentParser (parsers.py)
```python
class DocumentParser:
    """Parse different document types"""

    @staticmethod
    def parse_pdf(file_path: str) -> str:
        """Extract text from PDF"""

    @staticmethod
    def parse_txt(file_path: str) -> str:
        """Read text file"""

    @staticmethod
    def parse_docx(file_path: str) -> str:
        """Extract text from DOCX"""

    def parse(self, file_path: str) -> str:
        """Auto-detect and parse document"""
```

### API Models (models.py)
```python
class DocumentChunk(BaseModel):
    doc_id: str
    chunk_id: int
    text: str
    metadata: dict

class IndexRequest(BaseModel):
    doc_id: str
    file_path: str
    metadata: dict = {}

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    chunks: list[DocumentChunk]
    scores: list[float]
```

### CLI Tool (cli.py)
```python
import click

@click.group()
def cli():
    pass

@cli.command()
@click.argument('path')
@click.option('--recursive', is_flag=True)
def index(path: str, recursive: bool):
    """Index documents from path"""
    # Parse files -> chunk -> send to /index endpoint

@cli.command()
@click.argument('query')
def search(query: str):
    """Search indexed documents"""
    # Call /search endpoint -> print results

@cli.command()
def list():
    """List all indexed documents"""

@cli.command()
def clear():
    """Clear all documents"""
```

---

## Chat App Integration

### RAG Client (app/services/rag_client.py)
```python
class RAGClient:
    """HTTP client for RAG service"""

    def __init__(self, base_url: str):
        self.base_url = base_url  # e.g., "http://gpu-server:8001"

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for relevant context"""

    async def index_document(self, file_path: str, metadata: dict):
        """Index a document"""
```

### Document Upload (app/routes/documents.py)
```python
from fastapi import APIRouter, UploadFile

router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.post("/upload")
async def upload_document(file: UploadFile, user: User = Depends(get_current_user)):
    """Upload and index document"""
    # Save file -> call RAG service /index

@router.get("/")
async def list_documents(user: User = Depends(get_current_user)):
    """List user's documents"""
```

### WebSocket Integration (app/routes/websocket.py)
```python
# In websocket_chat_endpoint, before LLM call:

# 1. Search for relevant context
rag_client = RAGClient(os.getenv("RAG_SERVICE_URL"))
context_chunks = await rag_client.search(user_message, top_k=3)

# 2. Build enhanced prompt with context
if context_chunks:
    context_text = "\n\n".join([c["text"] for c in context_chunks])
    system_message = {
        "role": "system",
        "content": f"Use this context to answer:\n\n{context_text}"
    }
    llm_messages.insert(0, system_message)

# 3. Send to LLM as usual
```

---

## Implementation Steps

### 1. GPU Server Setup
```bash
cd /workspace/chat
uv init rag-service
cd rag-service
uv add fastapi uvicorn sentence-transformers qdrant-client pymupdf python-docx click torch
```

### 2. Create Core Files
- `embedder.py` - Load model on startup
- `vector_store.py` - Initialize Qdrant collection
- `chunker.py` - Implement chunking logic
- `parsers.py` - PDF/txt/docx parsers
- `models.py` - Pydantic schemas

### 3. Build FastAPI App (main.py)
```python
from fastapi import FastAPI
from embedder import Embedder
from vector_store import VectorStore

app = FastAPI()
embedder = Embedder()  # Loads on startup (GPU)
vector_store = VectorStore()

@app.post("/index")
async def index_document(request: IndexRequest):
    # Parse -> chunk -> embed -> store

@app.post("/search")
async def search(request: SearchRequest):
    # Embed query -> vector search -> return chunks
```

### 4. Build CLI
```python
# cli.py
@cli.command()
def index(path: str):
    files = find_files(path)
    for file in files:
        response = requests.post(f"{RAG_URL}/index", json={
            "doc_id": file.name,
            "file_path": str(file),
        })
```

### 5. Chat App Integration
- Add `RAGClient` service
- Modify WebSocket to inject context
- Add upload UI button
- Add `/api/documents/upload` endpoint

---

## Configuration

### GPU Server (.env)
```env
MODEL_NAME=sentence-transformers/all-mpnet-base-v2
QDRANT_PATH=./data/qdrant
UPLOAD_DIR=./data/uploads
HOST=0.0.0.0
PORT=8001
```

### Chat App (.env addition)
```env
RAG_SERVICE_URL=http://gpu-server-ip:8001
```

---

## Testing Flow

1. Start GPU server: `cd rag-service && uv run uvicorn main:app --host 0.0.0.0 --port 8001`
2. Index documents: `uv run python cli.py index /path/to/docs --recursive`
3. Test search: `uv run python cli.py search "query"`
4. Start chat app: `cd ai-chat-app && uv run python main.py`
5. Chat and verify context injection

---

## Performance Estimates (3090 RTX)

- **Embedding speed**: ~2000 docs/sec
- **Search latency**: <50ms for 1000 docs
- **Chunk size**: 500 tokens = ~375 words
- **Memory usage**: ~2GB VRAM (model) + 1GB (vectors for 1000 docs)

---

## Future Enhancements (post-hackathon)

- Re-ranking with cross-encoder
- Hybrid search (keyword + semantic)
- Multi-user document isolation
- Document versioning
- Metadata filtering
