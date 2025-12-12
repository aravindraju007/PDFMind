import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.pdf_parser import extract_text
from services.chunk import chunk_text
from services.embedder import embed_chunks
from services.vector_db import (
    init_chroma,
    store_chunks,
    query_similar,
    list_documents,
    delete_document_by_id
)
from services.llm import ask_ollama
from services.utils import save_uploaded_file

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

app = FastAPI(title="Multi-PDF Chat Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize chroma client/collection on startup
@app.on_event("startup")
def startup_event():
    logger.info("Starting up — initializing Chroma...")
    init_chroma(persist_directory=config.CHROMA_PERSIST_DIR)
    logger.info("Chroma initialized at %s", config.CHROMA_PERSIST_DIR)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save the raw uploaded file to disk (optional but helpful)
    saved_path = save_uploaded_file(file)
    logger.info("Saved uploaded file to %s", saved_path)

    # Extract text from PDF
    try:
        text_pages = extract_text(saved_path, return_pages=True)
    except Exception as e:
        logger.exception("Failed to extract PDF text")
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {e}")

    # Create chunks per page (we include filename and page metadata)
    filename = os.path.basename(saved_path)
    chunks_with_meta = []
    for page_idx, page_text in enumerate(text_pages):
        if not page_text or not page_text.strip():
            continue
        page_chunks = chunk_text(page_text)
        for c in page_chunks:
            meta = {"filename": filename, "page": page_idx}
            chunks_with_meta.append((c, meta))

    if len(chunks_with_meta) == 0:
        return JSONResponse({"message": "No textual content detected in PDF", "chunks": 0})

    # Embeddings
    try:
        chunks = [c for c, m in chunks_with_meta]
        embeddings = embed_chunks(chunks)
    except Exception as e:
        logger.exception("Embedding generation failed")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # Store in vector DB
    try:
        stored_ids = store_chunks(chunks, embeddings, [m for _, m in chunks_with_meta])
    except Exception as e:
        logger.exception("Failed to store chunks")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    return {"message": "PDF uploaded & indexed", "chunks": len(chunks), "ids_sample": stored_ids[:5]}

@app.post("/ask")
async def ask_question(payload: dict):
    question = payload.get("question")
    top_k = payload.get("top_k", 5)
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    # retrieve context
    try:
        context_items = query_similar(question, n=top_k)
    except Exception as e:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    combined_context = "\n\n".join([f"[{i['metadata'].get('filename','?')} - p{ i['metadata'].get('page','?') }]\n{i['document']}" for i in context_items])
    # call LLM
    try:
        answer = ask_ollama(question, combined_context)
    except Exception as e:
        logger.exception("LLM call failed")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

    # return structured response: answer + sources
    sources = [{"snippet": item["document"][:500], "metadata": item["metadata"]} for item in context_items]
    return {"answer": answer, "sources": sources}

@app.get("/documents")
def get_documents():
    docs = list_documents()
    return {"documents": docs}

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    ok = delete_document_by_id(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": doc_id}
