"""
Each guard runs on every request and answers one question:
  1. check_question   - should this reach the pipeline at all?
  2. check_retrieval  - did we find evidence good enough to answer from?
  3. check_citations  - is every claimed quote really in the chunk it cites?

"""

import re

from rag.chunk import Chunk

# The classic prompt-injection phrasings. Retrieved text is data, not
# instructions - but the cheapest place to stop an attack is the front door.
INJECTION = re.compile(
    r"ignore (the |all )?(previous|above) instructions|you are now|reveal your (system )?prompt",
    re.I,
)

MIN_SCORE = 0.10  # cosine similarity below this means "nothing relevant exists"


def check_question(question: str) -> str | None:
    """Returns a refusal message, or None if the question may proceed."""
    if not 3 <= len(question) <= 500:
        return "Please ask a question between 3 and 500 characters."
    if INJECTION.search(question):
        return "That looks like an attempt to change my instructions, so I did not run it."
    return None


def check_retrieval(results: list[tuple[Chunk, float]]) -> str | None:
    """A vector store always returns k results - even for nonsense. The score
    floor turns 'best of nothing' into an honest refusal instead of a guess."""
    if not results or results[0][1] < MIN_SCORE:
        return "I could not find anything in the indexed papers about that."
    return None


def check_citations(citations: list[dict], results: list[tuple[Chunk, float]]) -> list[dict]:
    """Keep only citations whose quote appears verbatim in the chunk they cite."""
    by_id = {chunk.id: chunk for chunk, _ in results}
    valid = []
    for citation in citations:
        chunk = by_id.get(citation.get("chunk_id"))
        quote = citation.get("quote", "")
        # Whitespace-insensitive: models reflow line breaks when quoting.
        if chunk and len(quote) > 10 and _squash(quote) in _squash(chunk.text):
            valid.append({**citation, "source": f"{chunk.title}, {chunk.section}, p.{chunk.page}"})
    return valid


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
