"""
chunker.py — Splits loaded documents into smaller overlapping chunks
suitable for embedding and retrieval.

Splitting is boundary-aware: it prefers to break on paragraph or sentence
boundaries rather than cutting blindly at a fixed character offset, so
chunks are less likely to end mid-sentence. Overlap between consecutive
chunks preserves context across boundaries that do still cut through
related content.
"""

import re
from dataclasses import dataclass

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE
from rag.document_loader import Document

# Matches paragraph breaks first, then sentence ends, as candidate split points.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """A piece of a document, sized for embedding.

    source: the originating document's filename, carried through from
        Document so retrieval results can be attributed back to a file.
    text: the chunk's text content.
    chunk_index: position of this chunk within its source document,
        used only for debugging/inspection, not for retrieval logic.
    """
    source: str
    text: str
    chunk_index: int


def _split_into_segments(text: str) -> list[str]:
    """Break text into paragraph-level segments, falling back to sentence
    splits for any segment that's still too large on its own.
    """
    paragraphs = _PARAGRAPH_BREAK.split(text)
    segments: list[str] = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= CHUNK_SIZE:
            segments.append(paragraph)
        else:
            # Paragraph itself exceeds chunk size; split on sentences instead.
            segments.extend(
                s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip()
            )

    return segments


def _pack_segments(segments: list[str]) -> list[str]:
    """Greedily combine segments into chunks close to CHUNK_SIZE, carrying
    the trailing CHUNK_OVERLAP characters of each chunk into the next so
    boundary context isn't lost.
    """
    chunks: list[str] = []
    current = ""

    for segment in segments:
        candidate = f"{current} {segment}".strip() if current else segment

        if len(candidate) <= CHUNK_SIZE:
            current = candidate
            continue

        if current:
            chunks.append(current)
            # Seed the next chunk with overlap from the end of this one.
            overlap_text = current[-CHUNK_OVERLAP:]
            current = f"{overlap_text} {segment}".strip()
        else:
            # Single segment longer than CHUNK_SIZE on its own; keep it
            # whole rather than splitting mid-word.
            chunks.append(segment)
            current = ""

    if current:
        chunks.append(current)

    return chunks


def chunk_document(document: Document) -> list[Chunk]:
    """Split a single Document into a list of Chunks."""
    segments = _split_into_segments(document.content)
    packed = _pack_segments(segments)
    return [
        Chunk(source=document.source, text=text, chunk_index=i)
        for i, text in enumerate(packed)
    ]


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Split a list of Documents into a flat list of Chunks."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks


if __name__ == "__main__":
    from rag.document_loader import load_documents

    docs = load_documents()
    all_chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} documents -> {len(all_chunks)} chunks")
    for c in all_chunks[:3]:
        print(f"  - {c.source} [chunk {c.chunk_index}] ({len(c.text)} chars)")
        print(f"    {c.text[:100]}...")