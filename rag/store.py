"""Step 4 - Vector store: index the vectors, search by similarity (Week 3: "Vector Databases").

A vector store does three things: hold the vectors, find the nearest ones to a
query fast, and persist to disk. FAISS is that as a library - no server to run.

We use a flat (brute force) index: at a thousand chunks, exact search is
instant. Approximate indexes like HNSW exist for the million-scale case, where
scanning every vector per query stops being an option; the interface stays the
same, which is why swapping the index type later is cheap.
"""

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from rag.chunk import Chunk
from rag.embed import embed

DIR = Path(__file__).parent.parent / "data" / "index"


def build(chunks: list[Chunk]) -> None:
    """Embed every chunk once and write the index + chunk data to disk."""
    vectors = embed([c.text for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])  # IP on unit vectors == cosine
    index.add(vectors)
    DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(DIR / "index.faiss"))
    (DIR / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks]))


def load() -> tuple[faiss.Index, list[Chunk]]:
    if not (DIR / "index.faiss").exists():
        raise FileNotFoundError("no index found - run: python ingest.py")
    index = faiss.read_index(str(DIR / "index.faiss"))
    chunks = [Chunk(**c) for c in json.loads((DIR / "chunks.json").read_text())]
    return index, chunks


def search(index: faiss.Index, chunks: list[Chunk], question: str, k: int = 4) -> list[tuple[Chunk, float]]:
    """The whole of retrieval: embed the question, take the k nearest chunks."""
    query = embed([question])
    scores, ids = index.search(query, k)
    return [(chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]
