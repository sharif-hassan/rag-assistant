"""
query_rewriter.py — Rewrites user questions into better retrieval queries.

A conversational question like "What is FastAPI?" embeds differently from
the documentation prose it's trying to retrieve. Rewriting it to something
like "FastAPI web framework overview features purpose" closes that semantic
gap and improves retrieval quality.

The original question is always preserved for answer generation — only
the retrieval step uses the rewritten query.
"""

from openai import OpenAI
from rag.config import OPENAI_API_KEY, OPENAI_CHAT_MODEL

_client = OpenAI(api_key=OPENAI_API_KEY)

_REWRITE_PROMPT = """You are a search query optimizer for a technical documentation system.

Rewrite the user's question into a concise search query that will retrieve
the most relevant documentation chunks. Focus on technical keywords and
concepts rather than conversational phrasing.

Rules:
- Return ONLY the rewritten query, nothing else
- Keep it under 15 words
- Remove filler words like "how do I", "what is", "can you explain"
- Preserve all technical terms exactly as written
- Do not answer the question, only rewrite it

Examples:
  "What is FastAPI?" -> "FastAPI web framework overview features purpose"
  "How do I set up a virtual environment?" -> "virtual environment setup Python venv activate"
  "What's the difference between async and sync?" -> "async def vs def path operations FastAPI difference"

User question: {question}
Rewritten query:"""


def rewrite_query(question: str) -> str:
    """Rewrite a conversational question into a better retrieval query.

    Falls back to the original question if rewriting fails, so the
    pipeline never breaks due to this step.
    """
    try:
        response = _client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": _REWRITE_PROMPT.format(question=question),
                }
            ],
            temperature=0,
            max_tokens=50,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else question
    except Exception:
        return question


if __name__ == "__main__":
    test_questions = [
        "What is FastAPI?",
        "How do I set up a virtual environment for FastAPI?",
        "What's the difference between async and sync path operations?",
        "How does FastAPI compare to Flask?",
        "What environment variables does FastAPI use?",
    ]

    print("Query rewriting test\n" + "=" * 40)
    for q in test_questions:
        rewritten = rewrite_query(q)
        print(f"Original:  {q}")
        print(f"Rewritten: {rewritten}")
        print()