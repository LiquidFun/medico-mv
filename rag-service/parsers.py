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
