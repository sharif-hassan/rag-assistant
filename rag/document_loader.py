"""
document_loader.py — Reads raw documents from disk and converts them into
a clean, uniform representation for the rest of the pipeline.

Markdown files are read directly since they're already plain text. Each
file becomes a Document, which tracks its source filename alongside the
content so later stages (chunking, retrieval) can preserve attribution
back to the original file.
"""

from dataclasses import dataclass
from pathlib import Path

from rag.config import DATA_RAW_DIR


@dataclass
class Document:
    """A single loaded document before chunking.

    source: filename it came from, used later for citing answers.
    content: the full raw text of the file.
    """
    source: str
    content: str


def load_documents(raw_dir: str = DATA_RAW_DIR) -> list[Document]:
    """Read all markdown files from raw_dir and return them as Documents.

    Skips empty files. Raises no error on an empty directory — returns
    an empty list instead, so callers can decide how to handle that case.
    """
    raw_path = Path(raw_dir)
    documents: list[Document] = []

    for file_path in sorted(raw_path.glob("*.md")):
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        documents.append(Document(source=file_path.name, content=content))

    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {DATA_RAW_DIR}")
    for doc in docs:
        print(f"  - {doc.source} ({len(doc.content)} chars)")