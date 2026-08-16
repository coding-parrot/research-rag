# From the notebooks to this repo

Where each cell of the two teaching notebooks ended up, and what changed on the
way to production. Useful as the bridge if the notebooks are Sessions 1-2 of a
cohort and this repo is Session 3.

## Session 1 - Vanilla RAG (LegalBot)

| Notebook | Cell(s) | Here | What changed |
|---|---|---|---|
| Download PDFs with `urllib` | 6 | `ingest/manifest.py`, `ingest/fetch.py` | Corpus is a versioned YAML manifest; downloads are sha256-pinned and content-addressed. A changed upstream file fails loudly instead of silently re-ingesting. |
| `PyPDFLoader` per page | 7-8 | `ingest/ocr/surya.py`, `ingest/normalize.py` | The notebook says scanned/multi-column PDFs are "out of scope today". This is that scope: Surya layout + reading order + tables, cached to JSON so it runs once per paper. |
| `RecursiveCharacterTextSplitter` | 11-12 | *(removed)* | Per your direction: section chunking only. |
| `MarkdownHeaderTextSplitter` + `HEADING_RE` | 13-19 | `ingest/headers.py`, `chunking/section.py` | The Title-Case regex ("not bulletproof") became a four-signal detector (PDF outline, Surya layout, numbered regex, font) merged by agreement, with a labelled eval scoring it. Oversized sections split with the header inherited; tiny ones merge forward. |
| `HuggingFaceEmbeddings` MiniLM | 21-22 | `embed/models.py` | Same model. Adds an on-disk embedding cache and a fingerprint so an index refuses to load under a different model. |
| Manual cosine demo | 24-25 | `embed/base.py` | `cosine_similarity` is the same five lines of numpy, now with tests. |
| FAISS + `save_local` | 30-34 | `index/stores.py` | Same FAISS flat index. Pickle deserialisation (`allow_dangerous_deserialization=True`) replaced with JSONL + npy, readable and safe to load. In-memory store is the tested oracle. |
| `as_retriever(k=4)` | 36-41 | `retrieve/retriever.py` | Top-k survives inside a bigger pipeline: hybrid BM25+dense with RRF fusion, dedup, rerank, per-document cap. |
| Grounded prompt with `[source: filename]` | 44-46 | `generate/prompts.py`, `guardrails/output_guard.py` | The three rules (ground, allow refusal, cite) survive. Free-text citations became structured `{chunk_id, quote}` validated mechanically: unverifiable citation formats are the single biggest change from the notebook. |
| Ollama `gemma2:2b` | 48-49 | `generate/client.py` | Kept as a provider option. Claude Opus 5 is the default; both sit behind one `LlmClient` protocol with a fake for tests. |
| LCEL chain | 51 | `pipeline.py` | The `|` composition became an explicit `Pipeline.ask()` with typed stages, because you cannot unit-test a pipe operator's error handling. |

## Session 2 - Improvements

| Notebook | Cell(s) | Here | What changed |
|---|---|---|---|
| `MultiQueryRetriever` + custom parser | 58-61 | `retrieve/rewrite.py` | Same idea, no LangChain. The defensive `QueryListParser` survives as `parse_query_list`. Falls back to vanilla on any error; the original query always leads. |
| HyDE from scratch | 63-65 | `retrieve/rewrite.py` | Same ten lines, plus: the hypothetical is *fused with* the original query rather than replacing it, and a too-short hypothetical falls back. |
| Cohere rerank | 67-69 | `retrieve/rerank.py` | Kept as an option, key from env only. Default is a local cross-encoder so the eval matrix costs nothing to sweep. **The notebook's hardcoded API key is the anti-pattern this repo's secrets handling exists to prevent.** |
| Input guard: `DOMAIN_TERMS` keyword overlap | 73-74 | `guardrails/input_guard.py` | Keyword overlap over-refuses ("how long do you keep my stuff?" has zero matches). Replaced by an embedding-centroid scope classifier, with false-refusal rate as a tracked metric. The injection regexes survive, extended, and now also scan *retrieved chunks*, the surface the notebook named but never checked. |
| Groundedness judge (gemma2 judging gemma2) | 76-77 | `eval/judge.py` | Judging moved out of the request path into evals, on a separate stronger model, with a human-agreement calibration. In the request path, the deterministic citation check does the work: it catches most fabrication for free. |
| `legalbot()` full chain | 79-81 | `pipeline.py` | Same shape: input guard -> retrieval -> generate -> output guard. Differences: typed `Answer`/`Decision` instead of a dict with a `blocked` bool, refusal as a first-class status, trace ids, token accounting. |
| *(absent)* | — | `eval/` | The notebook has zero evals: three demo questions printed to stdout. Here: golden set, adversarial set, header labels, deterministic CI gates, judge tier, committed results. |
