from sentence_transformers import SentenceTransformer
from typing import List

# load model once
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = SentenceTransformer(MODEL_NAME)

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Return list of embeddings (as lists of floats) for the chunks.
    """
    embs = _model.encode(chunks, show_progress_bar=False)
    # ensure lists not numpy arrays
    return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embs]

def embed_query(query: str) -> List[float]:
    e = _model.encode([query])[0]
    return e.tolist() if hasattr(e, "tolist") else list(e)
