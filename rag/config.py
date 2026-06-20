"""
config.py — Central configuration loaded from environment variables.

Why this exists:
  Every other module imports from here. Nothing hardcodes model names,
  paths, or API keys inline. This is what makes the system easy to reconfigure
  and easy to explain in interviews: "all config lives in one place."
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Reads .env file into os.environ


# ── LLM Backend ────────────────────────────────────────────────────────────────
# "openai" → cloud API (requires internet + key)
# "ollama" → local Llama 3 via Ollama (Phase 2, offline-capable)
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "openai")

# ── OpenAI ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ── Ollama (Phase 2) ────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

# ── Vector Store ────────────────────────────────────────────────────────────────
# ChromaDB persists to disk at this path. Offline-friendly by design.
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

# ── Chunking ────────────────────────────────────────────────────────────────────
# CHUNK_SIZE: how many characters per chunk.
# CHUNK_OVERLAP: how many characters the next chunk shares with the previous.
# Overlap helps prevent answers from being cut off at chunk boundaries.
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# ── Retrieval ───────────────────────────────────────────────────────────────────
# How many chunks to retrieve and pass to the LLM as context.
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))

# ── Paths ───────────────────────────────────────────────────────────────────────
DATA_RAW_DIR: str = "./data/raw"
DATA_PROCESSED_DIR: str = "./data/processed"
EVAL_DIR: str = "./evaluation"
