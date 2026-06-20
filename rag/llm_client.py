"""
llm_client.py — Generates answers from retrieved context chunks.

This is the only file in the pipeline that knows which LLM backend is
active. Everything else calls generate_answer() and gets text back,
regardless of whether that text came from OpenAI or a local Ollama model.

Swapping backends requires changing one environment variable (LLM_BACKEND)
and nothing else in the codebase.
"""

from openai import OpenAI

from rag.config import (
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
)

# OpenAI client — only used when LLM_BACKEND=openai
_openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Ollama client — reuses the OpenAI SDK since Ollama exposes an
# OpenAI-compatible API endpoint. Only the base_url and api_key differ.
_ollama_client = OpenAI(
    base_url=f"{OLLAMA_BASE_URL}/v1",
    api_key="ollama",  # Ollama doesn't require a real key; this is a placeholder.
)

_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about FastAPI
based strictly on the provided documentation excerpts.

Rules:
- Only use information from the provided context to answer.
- If the context does not contain enough information to answer, say so clearly.
- Always cite which document(s) your answer comes from.
- Be concise and accurate.
"""


def _build_prompt(question: str, chunks: list[dict]) -> str:
    """Assemble the context chunks and question into a single user message."""
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(f"[{i}] Source: {chunk['source']}\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    return f"Context:\n\n{context}\n\nQuestion: {question}"


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Generate an answer grounded in the retrieved chunks.

    Selects the backend based on LLM_BACKEND config value.
    Raises ValueError if an unsupported backend is specified.
    """
    prompt = _build_prompt(question, chunks)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    if LLM_BACKEND == "openai":
        response = _openai_client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content

    if LLM_BACKEND == "ollama":
        response = _ollama_client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content

    raise ValueError(
        f"Unsupported LLM_BACKEND: {LLM_BACKEND!r}. "
        "Expected 'openai' or 'ollama'."
    )


if __name__ == "__main__":
    from rag.config import TOP_K_RESULTS
    from rag.embedder import VectorStore

    store = VectorStore()
    question = "How does FastAPI use Python type hints?"
    chunks = store.query(question, top_k=TOP_K_RESULTS)

    print(f"Question: {question}\n")
    print(f"Retrieved {len(chunks)} chunks from: {[c['source'] for c in chunks]}\n")
    print("Answer:")
    print(generate_answer(question, chunks))