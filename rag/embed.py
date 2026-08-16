"""Step 3 - Embedding: text becomes a vector (Week 3: "Embedding Models").

An embedding model maps text to a fixed-length vector so that similar meanings
land near each other. "Near" has a precise definition: cosine similarity. We
normalise every vector to length 1, which turns cosine into a plain dot product.

The model is a contract: the same model must embed the chunks at index time and
the query at question time. Swap the model without re-indexing and every search
silently returns nonsense.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions, runs locally

_model = None  # loaded once per process; loading takes seconds, encoding is fast


def embed(texts: list[str]) -> np.ndarray:
    """List of texts -> matrix of unit-length vectors, one row per text."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
