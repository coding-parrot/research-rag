"""Step 3 - Embedding: text becomes a vector (Week 3: "Embedding Models").

An embedding model maps text to a fixed-length vector so that similar meanings
land near each other. "Near" has a precise definition: cosine similarity. OpenAI's
embeddings come back already normalised to length 1, which turns cosine into a
plain dot product - exactly what the FAISS inner-product index computes.

The model is a contract: the same model must embed the chunks at index time and
the query at question time. Swap the model without re-indexing and every search
silently returns nonsense.
"""

from pathlib import Path

import numpy as np
from openai import OpenAI

MODEL = "text-embedding-3-large"  # 3072 dimensions
BATCH = 128  # texts per API request

_client = None


def api_key() -> str:
    """OPENAI_API_KEY from the .env file next to the project."""
    for line in (Path(__file__).parent.parent / ".env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("put OPENAI_API_KEY=... in .env")


def embed(texts: list[str]) -> np.ndarray:
    """List of texts -> matrix of unit-length vectors, one row per text."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=api_key())

    vectors = []
    for start in range(0, len(texts), BATCH):
        response = _client.embeddings.create(model=MODEL, input=texts[start : start + BATCH])
        vectors.extend(item.embedding for item in response.data)
    return np.array(vectors, dtype="float32")
