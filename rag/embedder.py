"""
embedder.py — Generates embeddings for chunks and persists them in
ChromaDB, the local vector store.

Two responsibilities are kept separate on purpose:
  1. embed_texts(): calls the embedding provider (OpenAI) and returns vectors.
  2. VectorStore: wraps ChromaDB for adding and querying those vectors.

Keeping them separate means swapping the embedding provider later only
requires changing embed_texts(), not anything that talks to ChromaDB.
"""

import chromadb
from openai import OpenAI

from rag.chunker import Chunk
from rag.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
)

_client = OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert a list of strings into a list of embedding vectors.

    Batched into a single API call rather than one call per text, since
    OpenAI's embedding endpoint accepts a list directly — this matters
    once you're embedding hundreds of chunks instead of a handful.
    """
    response = _client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection.

    Persisting to disk (rather than running in-memory) is what makes the
    offline story real: once chunks are embedded and stored, querying
    them requires no further OpenAI calls and no internet connection.
    """

    def __init__(self) -> None:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and store a batch of chunks. Each chunk gets a stable,
        unique ID derived from its source and index so re-running
        ingestion on the same files doesn't create duplicate entries.
        """
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)
        ids = [f"{c.source}::{c.chunk_index}" for c in chunks]
        metadatas = [{"source": c.source, "chunk_index": c.chunk_index} for c in chunks]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, question: str, top_k: int) -> list[dict]:
        """Embed a question and return the top_k most similar chunks.

        Returns a list of dicts with 'text', 'source', and 'distance'
        (lower distance = more similar) so callers don't need to know
        anything about ChromaDB's internal result format.
        """
        question_embedding = embed_texts([question])[0]
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=top_k,
        )

        matches = []
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            matches.append(
                {
                    "text": text,
                    "source": metadata["source"],
                    "distance": distance,
                }
            )
        return matches


if __name__ == "__main__":
    from rag.chunker import chunk_documents
    from rag.document_loader import load_documents

    docs = load_documents()
    chunks = chunk_documents(docs)

    store = VectorStore()
    print(f"Embedding and storing {len(chunks)} chunks...")
    store.add_chunks(chunks)
    print("Done.")

    test_query = "How does FastAPI use Python type hints?"
    print(f"\nTest query: {test_query!r}")
    for match in store.query(test_query, top_k=3):
        print(f"  [{match['distance']:.4f}] {match['source']}: {match['text'][:80]}...")