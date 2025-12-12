# DocuraAI
A web app where users can, upload multiple PDF files.  Ask questions about all PDFs combined  get AI answers, citations, summaries, tables, insights, etc.  Do semantic search inside all PDFs,  optionally, save chat history, share links, export answers.

# DocuraAI — Zero-Cost Multi-PDF Chat App Chat with Multiple PDFs Locally using FastAPI · ChromaDB · Sentence-Transformers · Ollama · Next.js

DocuraAI is a fully local, privacy-first, zero-costPDF AI assistant.Upload multiple PDFs → ask natural language questions → get AI answers *with citations*, powered entirely by local open-source models.No OpenAI fees, no cloud vector DB, no GPU required.

##  Core Features

-  Upload multiple PDFs    
- Fast local embeddings using MiniLM.
- Semantic search over all documents via ChromaDB  
- Local LLM answering** using Ollama (Llama 3, Mistral, Phi-3, etc.)  
- Citations (filename + page) included in answers  
- 100% offline, data never leaves your machine  
- Modern UI with Next.js + Tailwind  
- Built for SaaS, but at zero infrastructure cost

## Tech Stack
### Backend
- FastAPI (Python)
- ChromaDB (local vector DB)
- Sentence-Transformers MiniLM** embeddings
- Ollama for LLM inference (Llama-3 by default)
- Uvicorn server

### Frontend
- Next.js 14 App Router
- TailwindCSS
- React
- 
## (RAG Pipeline)

* Extract text from each PDF

* Chunk into 300–400 word sections

* Convert chunks → embeddings via MiniLM

+ Store embeddings + metadata in ChromaDB

- User asks question → embed question

* Retrieve top-K relevant chunks

* nFeed context into Ollama LLM

* Return final answer + citations

* All done locally, with zero cloud cost.

