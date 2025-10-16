from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from app.services.rag_client import RAGClient
from app.services.auth import get_current_user
from app.models import User

router = APIRouter(prefix="/api/documents", tags=["documents"])
rag_client = RAGClient()


@router.get("/{doc_id}/pdf")
async def get_document_pdf(
    doc_id: str,
    user: User = Depends(get_current_user)
):
    """Proxy PDF from RAG service (authenticated)"""
    try:
        pdf_content = await rag_client.get_document_file(doc_id)
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"inline; filename={doc_id}.pdf",
                "Content-Type": "application/pdf"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Document not found: {str(e)}")
