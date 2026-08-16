# research-rag

Section-aware RAG over the [InterviewReady AI-engineering research library](https://github.com/InterviewReady/ai-engineering-resources).
Ask questions about the papers; get answers with citations that are mechanically
verified before they ship.

The design premise: **chunks are sections.** `1. Introduction`, `2. Method` and
`3. Experiments` become separate chunks, delimited by the paper's own headings.
Everything else in the system exists to make that one strategy trustworthy:
Surya OCR recovers reading order and layout on two-column PDFs, a layered detector
finds the headings, an eval suite measures whether it worked, and guardrails make
sure a wrong answer becomes a refusal rather than a hallucination.

## Pipeline

```
corpus.yaml -> fetch (sha256-pinned) -> Surya OCR (cached) -> normalize
    -> header detection (outline > layout > regex > font)
    -> section chunks (oversized split, undersized merged)
    -> embed (MiniLM) -> FAISS + BM25

ask(question)
    -> input guard        length / injection / secrets / scope
    -> retrieve           hybrid dense+BM25, RRF fusion, dedup, rerank, per-doc cap
    -> retrieval guard    relevance floor, injection scan on retrieved text
    -> generate           Claude Opus 5, structured {answer, citations}
    -> output guard       every citation checked: chunk_id retrieved? quote verbatim?
                          fail -> one regeneration -> typed refusal
```

Every stage returns typed `Decision` values instead of raising or mutating, so a
refusal always tells you which rule fired and why.

## Quick start

```bash
make setup-full        # venv + all dependencies (Surya, FAISS, sentence-transformers)
cp .env.example .env   # add ANTHROPIC_API_KEY, or use `ant auth login`

make ingest            # fetch PDFs, OCR, detect headers, chunk (slow first run, cached after)
make index             # embed and build the indexes
.venv/bin/rag ask "How does LoRA reduce trainable parameters?"
```

No API key? Set `generate.provider: ollama` in a config file and run against a
local model. The pipeline, guardrails and evals work identically.

## Evals

```bash
make eval        # deterministic metrics, free, gates CI
make eval-judge  # + LLM-judged faithfulness and correctness (spends tokens)
make headers     # header-detection precision/recall against hand labels
```

Three tiers:

1. **Deterministic, every PR**: recall@k, MRR, nDCG, context precision, citation
   validity, refusal precision/recall, false-refusal rate. Hard thresholds in
   `eval` config.
2. **LLM judge, on demand**: faithfulness and answer correctness, graded by a
   separate judge model with a reported human-agreement calibration.
3. **Header detection**: boundary precision/recall/F1 against hand-labelled
   section lists in `evals/headers/labels.yaml`. With one chunking strategy,
   this metric *is* chunk quality.

The golden set (`evals/golden/golden.yaml`) is model-bootstrapped and every item
is `reviewed: false` until a human verifies it against the paper. **Unreviewed
items never gate CI** — the runner refuses to enforce thresholds on them.

## Repository layout

```
src/rag/
  domain.py          frozen dataclasses, no dependencies; chunk ids are position-addressed and deterministic
  config.py          one hashable config drives every stage; run manifests record it
  ingest/            manifest -> fetch -> OCR (Surya, cached) -> normalize -> headers
  chunking/          SectionChunker: pure function, property-tested invariants
  embed/ index/      protocol + real + fake for embeddings, FAISS/in-memory, BM25
  retrieve/          RRF fusion, dedup, per-doc cap, multi-query, HyDE, rerankers
  generate/          LlmClient protocol (Anthropic / Ollama / fake), answerer with
                     enforced citations
  guardrails/        input / retrieval / output rules returning typed Decisions
  eval/              datasets, deterministic metrics, judge, runner
  app.py             composition root: the only module that picks implementations
  cli.py             rag ingest | index | ask | eval | headers | info

tests/               unit + property tests; fakes for OCR, embeddings, LLM;
                     no network, no weights, no tokens
corpus/corpus.yaml   the corpus: urls + pinned sha256 digests; PDFs never committed
evals/               golden set, adversarial items, header labels, results
```

## Design decisions worth knowing

- **Offsets are the source of truth.** Chunks and headings index into one
  normalised text; any provenance claim is checkable by slicing.
- **The relevance floor reads the dense cosine score**, not fused/reranked
  scores — RRF and cross-encoder outputs are not calibrated, and thresholding
  them is meaningless.
- **Citations are ids + verbatim quotes**, validated against the retrieved set.
  `[source: file.pdf]`-style free-text citations are unverifiable by design.
- **Retrieved text is untrusted.** The injection scan runs over chunks, not just
  the query; flagged chunks are quarantined with an explicit notice, not dropped
  (they may still be the correct answer).
- **Everything heavy is behind a protocol with a fake.** The full test suite
  runs in seconds on a laptop with nothing installed beyond `[dev]`.

## Licences

Papers are fetched from arXiv at ingest time under each paper's own licence and
are never redistributed. Surya's model weights are free for research and
personal use and for startups under $5M funding/revenue; commercial use beyond
that needs a licence from Datalab.
