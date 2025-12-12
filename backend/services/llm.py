import ollama
from typing import Optional
import config

def ask_ollama(question: str, context: str, system_prompt: Optional[str] = None) -> str:
    """
    Calls local ollama chat model with a single prompt combining context and user question.
    Ollama Python client returns a dict-like response; adapt as needed to your ollama version.
    """
    # assemble prompt - encourage model to use only context
    system = system_prompt or "You are a helpful assistant specialized in answering questions using only the provided document context. Cite sources with filename and page."
    prompt = f"""{system}

CONTEXT:
{context}

QUESTION:
{question}

Provide a concise answer. If the answer cannot be found in the context, say 'I don't know'. Include brief citations like [filename - pX].
"""
    # Use ollama.chat which streams in some versions — here we call simple chat
    resp = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    # Different ollama versions return different shapes; handle common case:
    if isinstance(resp, dict):
        # Some versions: {'message': {'content': '...'}}
        if "message" in resp and isinstance(resp["message"], dict):
            return resp["message"].get("content", "")
        # else maybe 'content' at top
        return resp.get("content", "") or str(resp)
    # fallback to string
    return str(resp)
