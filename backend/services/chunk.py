from typing import List

def chunk_text(text: str, max_len: int = 400) -> List[str]:
    """
    Simple word-based chunker. max_len is words per chunk.
    Returns list of chunk strings.
    """
    words = text.split()
    chunks = []
    current = []
    for w in words:
        current.append(w)
        if len(current) >= max_len:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks
