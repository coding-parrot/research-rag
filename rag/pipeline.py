"""The pipeline: every step in order, on one screen.

    question -> guard -> retrieve -> guard -> generate -> guard -> answer

Read this file first. Each line here is one lesson from Week 3, and each import
is the file that teaches it.
"""

from dataclasses import dataclass, field

from rag import generate, guards, store


@dataclass
class Answer:
    answer: str
    citations: list[dict] = field(default_factory=list)
    refused: bool = False


def ask(question: str) -> Answer:
    # Guard 1: is the question safe and sane?
    if refusal := guards.check_question(question):
        return Answer(answer=refusal, refused=True)

    # Retrieve: the k most similar chunks to the question.
    index, chunks = store.load()
    results = store.search(index, chunks, question)

    # Guard 2: is the evidence good enough to answer from?
    if refusal := guards.check_retrieval(results):
        return Answer(answer=refusal, refused=True)

    # Generate: answer + citations, from the passages only.
    raw = generate.generate(question, results)

    # Guard 3: keep only citations whose quotes verify against their chunks.
    citations = guards.check_citations(raw["citations"], results)
    if not citations:
        return Answer(
            answer="I could not produce an answer with verifiable citations, so I would rather refuse than guess.",
            refused=True,
        )
    return Answer(answer=raw["answer"], citations=citations)
