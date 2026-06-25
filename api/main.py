"""
main.py — FastAPI backend that exposes the RAG pipeline as a REST API.

Three endpoints:
  GET  /health   — liveness check
  POST /ask      — run a question through the RAG pipeline
  GET  /corpus   — list documents currently ingested

The RAG pipeline is instantiated once at startup and reused across
requests. On first startup, the vector store is automatically populated
from the corpus in data/raw/ if it's empty.
"""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.config import DATA_RAW_DIR
from rag.pipeline import RAGPipeline


# ── Auto-ingestion ────────────────────────────────────────────────────────────
def _ensure_corpus_ingested() -> None:
    """Ingest documents if the vector store is empty.

    Runs automatically on startup so the system works on first deploy
    without a separate ingestion step.
    """
    import chromadb
    from rag.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)

    if collection.count() == 0:
        print("Vector store empty — running ingestion...")
        from rag.document_loader import load_documents
        from rag.chunker import chunk_documents
        from rag.embedder import VectorStore

        docs = load_documents()
        chunks = chunk_documents(docs)
        store = VectorStore()
        store.add_chunks(chunks)
        print(f"Ingested {len(chunks)} chunks.")
    else:
        print(f"Vector store ready ({collection.count()} chunks).")


# ── Lifespan: instantiate pipeline once at startup ────────────────────────────
pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    _ensure_corpus_ingested()
    pipeline = RAGPipeline()
    yield
    pipeline = None


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Assistant API",
    description="FastAPI documentation Q&A powered by RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class ChunkResult(BaseModel):
    text: str
    source: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks: list[ChunkResult]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Run a question through the RAG pipeline and return a grounded answer."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    result = pipeline.ask(request.question.strip(), top_k=request.top_k)
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        chunks=[ChunkResult(**c) for c in result["chunks"]],
    )


@app.get("/corpus")
def corpus():
    """List the documents currently in the corpus."""
    try:
        files = sorted(
            f for f in os.listdir(DATA_RAW_DIR) if f.endswith(".md")
        )
    except FileNotFoundError:
        files = []
    return {"documents": files, "count": len(files)}