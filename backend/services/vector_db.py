import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from .embedder import embed_query
import os

_client = None
_collection = None

def init_chroma(persist_directory: str = "./chroma"):
    global _client, _collection
    os.makedirs(persist_directory, exist_ok=True)
    _client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
    # one collection holds all docs/chunks; metadata stores filename + page + optional doc_id
    _collection = _client.get_or_create_collection(name="documents", metadata={"hnsw:space": "cosine"})
    return _collection

def store_chunks(documents: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]):
    """
    Add chunks into Chroma collection.
    Returns list of ids stored.
    """
    global _collection
    if _collection is None:
        raise RuntimeError("Chroma collection not initialized")
    # generate ids - use unique id strategy with length of collection
    base = _collection.count() or 0
    ids = [f"chunk_{base + i}" for i in range(len(documents))]
    _collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
    return ids

def query_similar(query: str, n: int = 5):
    """
    Returns list of top-n results as list of dict:
    {"id":..., "document":..., "metadata":..., "distance": ...}
    """
    global _collection
    if _collection is None:
        raise RuntimeError("Chroma not initialized")
    q_emb = embed_query(query)
    res = _collection.query(query_embeddings=[q_emb], n_results=n, include=["documents", "metadatas", "distances", "ids"])
    docs = []
    # res dict structure: keys are lists indexed by queries; we use first query -> index 0
    for i in range(len(res["ids"][0])):
        docs.append({
            "id": res["ids"][0][i],
            "document": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i]
        })
    return docs

def list_documents():
    """
    Return list of unique filenames and counts.
    """
    global _collection
    if _collection is None:
        return []
    all_meta = _collection.get(include=["metadatas", "ids"])
    metas = all_meta.get("metadatas", [])
    # metas is list of metadata dicts
    files = {}
    for m in metas:
        if not m:
            continue
        fn = m.get("filename", "unknown")
        files.setdefault(fn, 0)
        files[fn] += 1
    return [{"filename": k, "chunks": v} for k, v in files.items()]

def delete_document_by_id(doc_id: str) -> bool:
    global _collection
    if _collection is None:
        return False
    try:
        # delete returns nothing; we attempt to delete by id
        _collection.delete(ids=[doc_id])
        return True
    except Exception:
        return False
