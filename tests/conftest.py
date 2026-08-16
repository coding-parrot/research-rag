"""Shared fixtures.

The `paper` fixture family builds documents through the real normalize -> detect ->
chunk path from fake OCR markup, so most tests exercise the actual pipeline code
rather than hand-assembled objects.
"""

from __future__ import annotations

import pytest

from rag.chunking.section import SectionChunker
from rag.config import ChunkConfig, Config, GuardrailConfig, HeaderConfig
from rag.domain import Chunk, NormalizedDocument
from rag.ingest.headers import HeaderDetector
from rag.ingest.normalize import NormalizationResult, normalize
from rag.ingest.ocr.fake import build_document

# A small paper with clean structure. Sections are long enough that none trip the
# min-chunk merge at the test config below.
PAPER_MARKUP = """\
[title] Selective Attention Networks

We introduce Selective Attention Networks, a family of models that adaptively \
prune attention heads at inference time. Our approach reduces compute by forty \
percent while preserving accuracy on standard benchmarks. This abstract block \
also stands in for the frontmatter chunk in tests.

# 1. Introduction

Attention mechanisms dominate modern sequence modelling, but most heads are \
redundant at inference time. Prior work prunes heads statically after training. \
We show that dynamic, input-conditioned pruning preserves accuracy at a fraction \
of the compute. Our contributions are threefold and described below in detail.

# 2. Method

We attach a lightweight gating network to every attention layer. The gate scores \
each head given the layer input, and heads below a learned threshold are skipped \
entirely. The gating network is trained jointly with a sparsity penalty. We call \
this mechanism selective head gating throughout the rest of the paper.

@page

# 3. Experiments

We evaluate on translation and summarisation benchmarks. Selective gating \
matches the dense baseline within 0.2 BLEU while skipping 40 percent of heads. \
Ablations show the learned threshold is critical: a fixed threshold loses a full \
BLEU point. We also measure wall-clock latency on commodity GPUs.

[table] Model | BLEU | Heads used
Dense | 27.4 | 100%
Selective | 27.2 | 60%

# 4. Conclusion

Dynamic head pruning is a practical inference-time optimisation. Future work \
includes extending selective gating to feed-forward layers and to mixture of \
expert models, where routing already provides a natural gating signal.
"""

# A paper whose Experiments section is long enough to force splitting under the
# test config, and whose tiny section 4 forces a merge.
LOPSIDED_MARKUP = (
    """\
[title] A Lopsided Paper

Abstract text that is long enough to be its own frontmatter chunk without being \
merged into anything that follows it in the pipeline under test configuration.

# 1. Introduction

A short but sufficient introduction section that comfortably clears the minimum \
chunk size configured in the tests, with a little padding text to be safe here.

# 2. Experiments

"""
    + " ".join(
        f"Sentence number {i} reports one more experimental result in detail." for i in range(120)
    )
    + """

# 3. Tiny

Too short.

# 4. Conclusion

A conclusion section that is long enough to stand alone as a chunk and absorb \
the tiny section before it when the merge policy runs over this document.
"""
)


@pytest.fixture()
def config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Test config: fake providers, small budgets, permissive thresholds.

    `paths.data` points at a fresh temp directory: the embed cache and index files
    are written wherever this config says, and a test run must never write into
    the repo working tree or read state a previous run left there.
    """
    return Config.model_validate(
        {
            "paths": {"data": str(tmp_path_factory.mktemp("data"))},
            "ocr": {"engine": "fake"},
            "chunk": {"max_chunk_tokens": 128, "part_overlap_tokens": 16, "min_chunk_chars": 80},
            "embed": {"provider": "fake", "model": "fake", "dimension": 32},
            "index": {"store": "inmemory", "bm25": True},
            "retrieve": {"strategy": "vanilla", "top_k": 4, "fetch_k": 10, "rerank": False},
            "generate": {"provider": "fake"},
            "guardrails": {"relevance_floor": 0.05, "scope_threshold": -1.0},
        }
    )


@pytest.fixture()
def header_config() -> HeaderConfig:
    return HeaderConfig()


@pytest.fixture()
def chunk_config(config: Config) -> ChunkConfig:
    return config.chunk


@pytest.fixture()
def guardrail_config(config: Config) -> GuardrailConfig:
    return config.guardrails


def make_normalized(markup: str, doc_id: str = "paper") -> NormalizationResult:
    return normalize(build_document(doc_id, markup))


def make_detected(
    markup: str, doc_id: str = "paper", header_config: HeaderConfig | None = None
) -> NormalizedDocument:
    """Markup -> normalized document with headings attached."""
    from dataclasses import replace

    result = make_normalized(markup, doc_id)
    detector = HeaderDetector(header_config or HeaderConfig())
    headings, _ = detector.detect(result)
    return replace(result.document, headings=headings)


def make_chunks(
    markup: str, doc_id: str = "paper", chunk_config: ChunkConfig | None = None
) -> tuple[Chunk, ...]:
    document = make_detected(markup, doc_id)
    chunker = SectionChunker(
        chunk_config
        or ChunkConfig(max_chunk_tokens=128, part_overlap_tokens=16, min_chunk_chars=80)
    )
    chunks, _ = chunker.chunk(document)
    return chunks


@pytest.fixture()
def paper_document() -> NormalizedDocument:
    return make_detected(PAPER_MARKUP)


@pytest.fixture()
def paper_chunks(chunk_config: ChunkConfig) -> tuple[Chunk, ...]:
    return make_chunks(PAPER_MARKUP, chunk_config=chunk_config)
