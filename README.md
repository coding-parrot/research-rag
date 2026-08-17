# research-rag

A RAG chatbot over 100 machine-learning papers, written to be read. One file per
pipeline step, in the same order as the Week 3 lecture: parse, chunk, embed,
store, retrieve, generate — plus the two things a demo never has: guardrails
and evals.

## Run it

```bash
git clone https://github.com/coding-parrot/research-rag
cd research-rag

python3.12 -m venv .venv
.venv/bin/pip install pymupdf faiss-cpu openai fastapi uvicorn pyyaml pytest numpy
echo "OPENAI_API_KEY=sk-..." > .env        # your own key; never commit this file

.venv/bin/python ingest.py                 # download + parse + embed the papers (one-time)
.venv/bin/python app.py                    # start the chatbot
```

Open http://127.0.0.1:8477 and ask: *"How does LoRA reduce trainable parameters?"*
(In PyCharm: point the interpreter at `.venv`, then right-click `app.py` → Run.)

## The map

```
papers.yaml        the corpus: 100 papers (the InterviewReady library + canon)
ingest.py          offline half: download -> parse -> chunk -> embed -> index

rag/parse.py       1. Parsing     PyMuPDF text; figures use the authors' own
                                  captions, a vision model only as fallback
rag/chunk.py       2. Chunking    split at section headings (numbered pattern +
                                  blacklist + monotonic numbers); split big, merge small
rag/embed.py       3. Embedding   OpenAI text-embedding-3-large (unit vectors)
rag/store.py       4. Vector DB   FAISS index: build, save, load, top-k search
rag/generate.py    5. Generation  passages + question -> {answer, citations} (JSON)
rag/guards.py      the three checks: question, evidence, citations
rag/pipeline.py    all of it wired together — read this file first

app.py             POST /ask + a chat page
eval.py            10 golden questions -> retrieval hit rate, grounded-citation rate
tests/test_rag.py  each test pins one lesson; runs with no key, no network
```

Models: `gpt-5.6-sol` answers questions, `gpt-5.6-luna` captions uncaptioned
figures (cached in `data/captions.json`), `text-embedding-3-large` embeds.
Swapping the embedding model invalidates the index — re-run `ingest.py`.

## The one idea that makes it trustworthy

The model must return citations as data — a chunk id and a verbatim quote — and
`guards.check_citations` verifies every quote against the actual chunk text.
A fabricated quote is dropped; an answer with no surviving citations becomes a
refusal. Hallucinated evidence cannot ship.

## Checks

```bash
.venv/bin/pytest tests -q      # unit tests, instant, no API key needed
.venv/bin/python eval.py       # end-to-end eval (uses the API)
                               # last run: 10/10 retrieval, 10/10 grounded citations
```

## Costs, roughly

One full ingest of 100 papers: ~2.7M embedding tokens (~$0.35) plus a few dozen
vision captions on first run (cached afterwards). Each question: one embedding
call plus one `gpt-5.6-sol` call over ~8 retrieved chunks.

## Papers

All 66 arXiv papers from the
[InterviewReady AI-engineering library](https://github.com/InterviewReady/ai-engineering-resources)
plus canonical additions (GPT-3, Scaling Laws, Chinchilla, LLaMA, DPO, ReAct,
DPR, ColBERT, Self-RAG, GPTQ, FlashAttention-2, ...). Papers are downloaded from
arXiv at ingest time and never committed; every download is title-checked
against its first page so a wrong ID cannot silently poison the corpus.
