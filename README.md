# RAG Assistant

A modular, offline-capable Retrieval-Augmented Generation (RAG) system designed for deployment in low-connectivity environments. Built with a backend-agnostic architecture that supports both cloud LLMs (OpenAI) and local open-source models (Llama 3 via Ollama).

Agriculture knowledge management is the primary domain case study, with the broader goal of making AI-grounded document retrieval practical in resource-constrained settings.

---

## Architecture Overview

```
Documents (PDF, TXT, DOCX)
        │
        ▼
  [ Ingestion & Chunking ]     ← document_loader.py, chunker.py
        │
        ▼
  [ Embedding ]                ← embedder.py  (OpenAI or local)
        │
        ▼
  [ Vector Store ]             ← vector_store.py  (ChromaDB, local disk)
        │
        ▼  (at query time)
  [ Retrieval ]                ← retriever.py
        │
        ▼
  [ Prompt Construction ]      ← prompt_builder.py
        │
        ▼
  [ LLM Generation ]           ← llm_client.py  ← ONLY file that knows which backend
        │
        ▼
  [ Source-Attributed Answer ] ← returned to CLI or API layer
```

**Key design decision:** `llm_client.py` is the only module that knows whether the system is using OpenAI or Ollama. Everything above it is backend-agnostic. Swapping models requires changing one environment variable.

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| Vector store | ChromaDB | Local disk persistence, no API key, offline-capable |
| Embeddings | OpenAI `text-embedding-3-small` | High quality, low cost |
| Generation (Phase 1) | OpenAI `gpt-4o-mini` | Fast, cheap, strong |
| Generation (Phase 2) | Llama 3 via Ollama | Fully offline, zero API cost |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Respects sentence/paragraph boundaries |
| CLI | Python + Rich | Clean terminal output for demos |

---

## Project Phases

| Phase | Scope | Status |
|---|---|---|
| **1** | Core RAG engine: ingest → embed → retrieve → answer (OpenAI backend, CLI) | 🔨 In progress |
| **2** | Local model backend via Ollama + Llama 3. Benchmark vs OpenAI | ⏳ Planned |
| **3** | Agriculture domain: curated corpus, domain eval, Sudan/low-connectivity framing | ⏳ Planned |
| **4** | Web app shell: FastAPI backend + React frontend | ⏳ Planned |
| **5** | Stretch: multilingual support, hybrid search, RAGAS evaluation | ⏳ Stretch |

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone <repo>
cd rag-assistant
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Ingest documents
python -m scripts.ingest --source data/raw/

# 4. Ask a question
python -m scripts.query "What are the best practices for sesame cultivation?"
```

---

## Repository Structure

```
rag-assistant/
├── rag/                    # Core library — all RAG logic lives here
│   ├── config.py           # Central config loaded from .env
│   ├── document_loader.py  # PDF/TXT/DOCX → raw text
│   ├── chunker.py          # Raw text → overlapping chunks
│   ├── embedder.py         # Chunks → embedding vectors
│   ├── vector_store.py     # ChromaDB read/write wrapper
│   ├── retriever.py        # Query → top-K relevant chunks
│   ├── prompt_builder.py   # Chunks + query → LLM prompt
│   └── llm_client.py       # LLM abstraction (OpenAI or Ollama)
├── scripts/
│   ├── ingest.py           # CLI: load documents into vector store
│   └── query.py            # CLI: ask questions against the index
├── evaluation/
│   └── eval_set.csv        # 10–15 Q&A pairs for Phase 1 evaluation
├── tests/
├── data/
│   ├── raw/                # Source documents (gitignored)
│   └── processed/          # Intermediate outputs if needed
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Design Principles

1. **Backend-agnostic core** — model swapping happens in one file
2. **Offline-first thinking** — ChromaDB needs no internet; Ollama (Phase 2) needs no API
3. **Source attribution** — every answer includes which document chunks it came from
4. **Incremental value** — each phase is independently deployable and resumé-worthy
