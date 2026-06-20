"""
cli.py — Interactive command-line interface for the RAG assistant.

Thin entry point: instantiates the pipeline once, then loops on user
input until the user quits. All RAG logic lives in pipeline.py.
"""

import sys
from rag.pipeline import RAGPipeline


def main() -> None:
    print("FastAPI Documentation Assistant")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    pipeline = RAGPipeline()

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            sys.exit(0)

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            sys.exit(0)

        result = pipeline.ask(question)

        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")
        print()


if __name__ == "__main__":
    main()