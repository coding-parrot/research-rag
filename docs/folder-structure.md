# Explaining the folder structure

Teach it in the same order as the Session 1 notebook. The pitch to the class:
**"Every cell in our notebook became a folder. The folder layout is the pipeline."**

## 1. The pipeline you already know, as folders

The notebook's six steps map one-to-one:

| Notebook step | Folder | What changed going to production |
|---|---|---|
| 1. Parse | `src/rag/ingest/` | PyPDFLoader became a pluggable engine (`ocr/`), plus header detection (`headers.py`) and text cleanup (`normalize.py`) |
| 2. Chunk | `src/rag/chunking/` | One strategy, done properly: sections, with split/merge policies |
| 3. Embed | `src/rag/embed/` | Same MiniLM, plus caching and a fingerprint so index and model can't drift apart |
| 4. Store | `src/rag/index/` | FAISS as before, plus BM25 for exact terms (hybrid) |
| 5. Retrieve | `src/rag/retrieve/` | top-k became: fuse dense+BM25, dedup, rerank, cap per paper |
| 6. Generate | `src/rag/generate/` | The prompt is versioned, the model is behind an interface, citations are structured data |

Then the two folders the notebook *didn't* have, which is the whole lesson:

| Folder | Why production forces it |
|---|---|
| `src/rag/guardrails/` | The notebook's `validate_input()` and groundedness check, grown into typed rules at three gates: input, retrieval, output |
| `src/rag/eval/` | The notebook had three print statements. This has a golden set, metrics that gate CI, and an LLM judge |

## 2. The files that aren't steps

Four files sit outside the pipeline. Explain them as the "rules of the house":

- `domain.py` — every noun in the system (Chunk, Answer, Citation, Decision).
  Frozen dataclasses, zero dependencies. *If you understand this file, you
  understand the system's vocabulary.*
- `config.py` — every knob, typed and hashable. A run can always say exactly
  which settings produced it.
- `pipeline.py` — the six steps wired together. ~80 lines. **Start reading here.**
- `app.py` — the only file allowed to choose concrete implementations
  (real OpenAI vs fake, FAISS vs in-memory). Everything else asks for
  interfaces. *This is why the tests run in one second with no API key.*

## 3. Trace one request live (the demo)

Ask the class to follow a question through the folders:

```
"How does LoRA work?"
  → pipeline.py            the conductor
  → guardrails/input_guard.py    is this safe and on-topic?
  → retrieve/retriever.py        embed query → FAISS + BM25 → fuse → top-4
  → guardrails/retrieval_guard.py  good enough evidence? scan chunks for injection
  → generate/answerer.py         build prompt → gpt-5.6-sol → {answer, citations}
  → guardrails/output_guard.py   every quote checked verbatim against its chunk
  → Answer                       with citations, decisions, token usage, trace id
```

Then show the same thing live: `POST /ask` and point at `decisions` in the
response — every gate the request passed through is recorded in the answer.

## 4. Everything else in one breath

- `src/rag/api/` — FastAPI wrapper: the pipeline gets an HTTP door and a chat page
- `tests/` — 367 tests, all running on fakes: no network, no models, no keys
- `corpus/corpus.yaml` — the library: 21 papers, sha256-pinned
- `evals/` — golden questions + hand-labelled section structure (the answer key)
- `data/` — gitignored: PDFs, caches, the index. Rebuildable from corpus.yaml
- `docs/notebook-mapping.md` — cell-by-cell: where each notebook cell ended up

## 5. The one-slide summary

> **Notebook → production is not a rewrite, it's a promotion.**
> Each cell became a folder. Each assumption became a typed check.
> Each "trust me" became a test. And two things the notebook never had —
> guardrails and evals — is where most of the production work lives.
