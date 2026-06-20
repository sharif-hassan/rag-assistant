"""
evaluate.py — Runs the eval set through the RAG pipeline and scores
each answer using an LLM-as-judge approach.

Each generated answer is scored 1-5 by GPT-4o-mini against the
reference answer. This is how production RAG systems are evaluated
when exact-match scoring isn't viable due to natural language variation.
"""

import json
import statistics
import time
from pathlib import Path

from openai import OpenAI

from rag.config import OPENAI_API_KEY, OPENAI_CHAT_MODEL
from rag.pipeline import RAGPipeline

EVAL_PATH = Path("./evaluation/eval_set.json")

_judge_client = OpenAI(api_key=OPENAI_API_KEY)

_JUDGE_PROMPT = """You are evaluating a RAG system that answers questions about FastAPI documentation.

Question: {question}
Reference answer: {reference}
Generated answer: {answer}

Score the generated answer from 1 to 5 using these criteria:
5 - Accurate, complete, and well-sourced. Covers all key points in the reference.
4 - Mostly accurate with minor omissions or imprecision.
3 - Partially correct but missing important information.
2 - Mostly incorrect or significantly incomplete.
1 - Wrong, irrelevant, or the system said it could not answer when it should have.

Respond with ONLY a single integer (1, 2, 3, 4, or 5). No explanation."""


def score_answer(question: str, reference: str, answer: str) -> int:
    """Ask GPT-4o-mini to score a generated answer against a reference."""
    response = _judge_client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": _JUDGE_PROMPT.format(
                    question=question,
                    reference=reference,
                    answer=answer,
                ),
            }
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        score = int(raw)
        return max(1, min(5, score))
    except ValueError:
        return 1


def run_evaluation(backend_label: str = "openai") -> None:
    """Run all eval questions through the pipeline and print a report."""
    eval_cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    pipeline = RAGPipeline()

    results = []
    print(f"Running evaluation ({len(eval_cases)} questions, backend: {backend_label})\n")

    for i, case in enumerate(eval_cases, 1):
        question = case["question"]
        reference = case["reference"]

        result = pipeline.ask(question)
        answer = result["answer"]
        sources = result["sources"]

        score = score_answer(question, reference, answer)
        results.append(score)

        status = "✓" if score >= 4 else "~" if score == 3 else "✗"
        print(f"[{i:02d}] {status} Score {score}/5 | {question[:60]}")
        print(f"      Sources: {', '.join(sources)}")

        time.sleep(0.5)

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS — Backend: {backend_label}")
    print("=" * 60)
    print(f"Questions:     {len(results)}")
    print(f"Average score: {statistics.mean(results):.2f} / 5.00")
    print(f"Median score:  {statistics.median(results):.1f} / 5.00")
    print(f"Score >= 4:    {sum(s >= 4 for s in results)}/{len(results)} ({100 * sum(s >= 4 for s in results) // len(results)}%)")
    print(f"Score <= 2:    {sum(s <= 2 for s in results)}/{len(results)} failures")
    print(f"Distribution:  {[results.count(s) for s in range(1, 6)]} (1s through 5s)")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    backend = sys.argv[1] if len(sys.argv) > 1 else "openai"
    run_evaluation(backend_label=backend)