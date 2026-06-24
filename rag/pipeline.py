"""
pipeline.py — Wires the RAG components into a single callable interface.

Nothing outside this file needs to know about document loading, chunking,
embedding, or LLM backends. Callers (CLI, web app, evaluation scripts)
just call ask() and get an answer back with its source attributions.

Query rewriting is applied before retrieval to improve semantic match
between conversational questions and documentation chunks.
"""

from rag.config import TOP_K_RESULTS
from rag.embedder import VectorStore
from rag.llm_client import generate_answer
from rag.query_rewriter import rewrite_query


class RAGPipeline:
    """End-to-end retrieval-augmented generation pipeline.

    Instantiated once and reused across queries so the VectorStore
    connection (and any future stateful components) aren't recreated
    on every question.
    """

    def __init__(self) -> None:
        self.store = VectorStore()

    def ask(self, question: str, top_k: int = TOP_K_RESULTS) -> dict:
        """Answer a question using retrieved context.

        Rewrites the question into a better retrieval query before
        searching, but always generates the answer against the original
        question so the response matches what the user actually asked.

        Returns a dict with:
          - answer: the generated response string
          - sources: deduplicated list of source filenames used
          - chunks: the raw retrieved chunks (useful for debugging/eval)
          - rewritten_query: the query actually used for retrieval
        """
        rewritten = rewrite_query(question)
        chunks = self.store.query(rewritten, top_k=top_k)
        answer = generate_answer(question, chunks)
        sources = list(dict.fromkeys(c["source"] for c in chunks))

        return {
            "answer": answer,
            "sources": sources,
            "chunks": chunks,
            "rewritten_query": rewritten,
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()

    questions = [
        "What is FastAPI?",
        "What environment variables does FastAPI use?",
        "What is the difference between async and sync in FastAPI?",
    ]

    for question in questions:
        print(f"Q: {question}")
        result = pipeline.ask(question)
        print(f"Rewritten: {result['rewritten_query']}")
        print(f"A: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("-" * 60)