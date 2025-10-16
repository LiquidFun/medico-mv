import os
from pathlib import Path
import fitz  # PyMuPDF
from docx import Document


class DocumentParser:
    """Parse different document types"""

    @staticmethod
    def parse_pdf(file_path: str) -> str:
        """Extract text from PDF"""
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    @staticmethod
    def parse_pdf_with_pages(file_path: str) -> list[dict]:
        """Extract text from PDF with page numbers"""
        doc = fitz.open(file_path)
        pages = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if text.strip():  # Only add pages with content
                pages.append({
                    "page_number": page_num,
                    "text": text
                })
        return pages

    @staticmethod
    def parse_txt(file_path: str) -> str:
        """Read text file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    @staticmethod
    def parse_docx(file_path: str) -> str:
        """Extract text from DOCX"""
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    def parse(self, file_path: str) -> str:
        """Auto-detect and parse document"""
        ext = Path(file_path).suffix.lower()

        if ext == '.pdf':
            return self.parse_pdf(file_path)
        elif ext == '.txt':
            return self.parse_txt(file_path)
        elif ext in ['.docx', '.doc']:
            return self.parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def parse_with_pages(self, file_path: str) -> list[dict]:
        """Parse document and return page-level data"""
        ext = Path(file_path).suffix.lower()

        if ext == '.pdf':
            return self.parse_pdf_with_pages(file_path)
        elif ext == '.txt':
            # TXT files don't have pages, treat as single page
            return [{"page_number": 1, "text": self.parse_txt(file_path)}]
        elif ext in ['.docx', '.doc']:
            # DOCX doesn't have clear pages, treat as single page
            return [{"page_number": 1, "text": self.parse_docx(file_path)}]
        else:
            raise ValueError(f"Unsupported file type: {ext}")
