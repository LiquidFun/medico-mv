from pydantic import BaseModel


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


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    uploaded_at: str
