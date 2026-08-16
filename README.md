# research-rag

A RAG chatbot over 21 ML papers, written to be read. One file per pipeline step,
in the same order as the Week 3 lecture: parse, chunk, embed, store, retrieve,
generate — plus the two things a demo never has: guardrails and evals.

## Run it

```bash
python3.12 -m venv .venv
.venv/bin/pip install pypdfium2 sentence-transformers faiss-cpu openai fastapi uvicorn pyyaml pytest
echo "OPENAI_API_KEY=sk-..." > .env

.venv/bin/python ingest.py                     # download + index the papers (one-time)
.venv/bin/uvicorn app:app --port 8477          # start the chatbot
```

Open http://127.0.0.1:8477 and ask: *"How does LoRA reduce trainable parameters?"*

## The map

```
papers.yaml        the corpus: 21 papers, one entry each
ingest.py          offline half: download -> parse -> chunk -> embed -> index

rag/parse.py       1. Parsing     PDF -> text per page
rag/chunk.py       2. Chunking    split at section headings; split big, merge small
rag/embed.py       3. Embedding   text -> unit vector (MiniLM, local)
rag/store.py       4. Vector DB   FAISS index: build, save, load, top-k search
rag/generate.py    5. Generation  chunks + question -> {answer, citations} (JSON)
rag/guards.py      the three checks: question, evidence, citations
rag/pipeline.py    all of it wired together — read this file first

app.py             POST /ask + a chat page
eval.py            10 golden questions -> retrieval hit rate, grounded-citation rate
tests/test_rag.py  each test pins one lesson; runs with no key, no network
```

## The one idea that makes it trustworthy

The model must return citations as data — a chunk id and a verbatim quote — and
`guards.check_citations` verifies every quote against the actual chunk text.
A fabricated quote is dropped; an answer with no surviving citations becomes a
refusal. Hallucinated evidence cannot ship.

## Checks

```bash
.venv/bin/pytest tests -q      # unit tests, instant, no API key
.venv/bin/python eval.py       # end-to-end eval (uses the API)
```
