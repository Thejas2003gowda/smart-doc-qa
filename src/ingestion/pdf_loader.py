from pypdf import PdfReader
from pathlib import Path


def load_pdf(file_path: str) -> list[dict]:
    """Extract text from PDF with page-level metadata."""
    reader = PdfReader(file_path)
    documents = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append({
                "content": text.strip(),
                "metadata": {
                    "source": Path(file_path).name,
                    "page": i + 1,
                    "total_pages": len(reader.pages)
                }
            })
    return documents