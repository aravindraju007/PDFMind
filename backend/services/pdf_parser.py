from typing import List
from pypdf import PdfReader

def extract_text(path: str, return_pages: bool = False):
    """
    Extract text from PDF file at path.
    If return_pages=True, return a list where each entry is page text.
    """
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    if return_pages:
        return pages
    return "\n".join(pages)
