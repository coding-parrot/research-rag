"""The composition root.

The only module that knows which concrete implementation fills each protocol.
Everything else takes its dependencies as constructor arguments, and the tests
compose their own graph out of fakes. If a module other than this one constructs a
Surya engine or an Anthropic client, that is a review comment.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.config import Config, Secrets
from rag.domain import Chunk
from rag.embed.base import Embedder
from rag.embed.models import build_embedder
from rag.errors import IndexError_
from rag.generate.answerer import Answerer
from rag.generate.client import LlmClient, build_client
from rag.guardrails.input_guard import InputGuard, ScopeClassifier
from rag.guardrails.output_guard import OutputGuard
from rag.guardrails.retrieval_guard import RetrievalGuard
from rag.index.base import ChunkStore, LexicalIndex, VectorStore
from rag.index.stores import Bm25Index, FaissVectorStore, InMemoryVectorStore
from rag.ingest.ocr.base import OcrEngine
from rag.observability import get_logger
from rag.pipeline import Pipeline
from rag.retrieve.rerank import build_reranker
from rag.retrieve.retriever import Retriever
from rag.retrieve.rewrite import build_transform

log = get_logger("app")


def build_ocr_engine(config: Config) -> OcrEngine:
    """OCR engine per config. The real engine is cache-wrapped; the fake is not.

    The fake replays fixture JSON from the OCR cache directory deterministically.
    Wrapping it in `CachedOcrEngine` would write fake-keyed duplicates back into
    the same directory it reads fixtures from, and would require hashing real PDFs
    for an engine that never opens one.
    """
    if config.ocr.engine == "fake":
        from rag.ingest.ocr.fake import FakeOcrEngine

        return FakeOcrEngine.from_fixtures(config.paths.ocr_cache)

    from rag.ingest.ocr.cached import CachedOcrEngine

    inner: OcrEngine
    if config.ocr.engine == "pypdfium":
        from rag.ingest.ocr.pdfium import PdfiumTextEngine

        inner = PdfiumTextEngine(config.ocr)
    else:
        from rag.ingest.ocr.surya import SuryaOcrEngine

        inner = SuryaOcrEngine(config.ocr)
    return CachedOcrEngine(inner, config.paths.ocr_cache, config.ocr)


@dataclass(slots=True)
class IndexBundle:
    """Everything the retriever needs, built once and persisted together."""

    store: ChunkStore
    vectors: VectorStore
    lexical: LexicalIndex | None
    embedder: Embedder

    def flush_embed_cache(self) -> None:
        """Persist any embeddings computed since the cache was loaded.

        Flushing is explicit, not automatic: the two places that embed corpus text
        (index build and the scope-classifier fit) call this, and retriever queries
        are never cached, so nothing else accumulates.
        """
        from rag.embed.models import CachedEmbedder

        if isinstance(self.embedder, CachedEmbedder):
            self.embedder.flush()


def build_index(
    config: Config, chunks: list[Chunk], *, embedder: Embedder | None = None
) -> IndexBundle:
    """Embed chunks and build fresh indexes."""
    embedder = embedder or build_embedder(
        config.embed, cache_dir=config.paths.index / "embed-cache"
    )
    store = ChunkStore(chunks)

    vectors: VectorStore
    if config.index.store == "faiss":
        vectors = FaissVectorStore(dimension=embedder.dimension)
    else:
        vectors = InMemoryVectorStore(dimension=embedder.dimension)

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    if chunks:
        vectors.add(ids, embedder.embed_documents(texts))

    lexical: LexicalIndex | None = None
    if config.index.bm25:
        lexical = Bm25Index()
        lexical.add(ids, texts)

    log.info(
        "index built",
        fields={"chunks": len(chunks), "store": config.index.store, "bm25": config.index.bm25},
    )
    return IndexBundle(store=store, vectors=vectors, lexical=lexical, embedder=embedder)


def save_index(bundle: IndexBundle, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    bundle.store.save(directory)
    bundle.vectors.save(directory)
    if bundle.lexical is not None:
        bundle.lexical.save(directory)
    # The index is only valid under the embedder that built it. Record its identity
    # so loading under a different model is an error instead of silent nonsense.
    # The fingerprint is deliberately written LAST: an interrupted rebuild in place
    # then leaves new stores under a stale (or absent) fingerprint, both of which
    # load_index rejects loudly. Writing it first would open a crash window where
    # OLD stores sit under the NEW fingerprint and load cleanly in the wrong space.
    (directory / "embedder.txt").write_text(bundle.embedder.fingerprint)
    bundle.flush_embed_cache()


def load_index(config: Config, *, embedder: Embedder | None = None) -> IndexBundle:
    directory = config.paths.index
    embedder = embedder or build_embedder(config.embed, cache_dir=directory / "embed-cache")

    fingerprint_path = directory / "embedder.txt"
    if not fingerprint_path.exists():
        # A missing fingerprint next to an existing index is a partial copy or an
        # interrupted save. Loading it under whatever the config now specifies is
        # exactly the silent wrong-space pairing the fingerprint exists to prevent,
        # so it is an error, not a skipped check.
        if (directory / ChunkStore.FILENAME).exists():
            raise IndexError_(
                f"index at {directory} has no recorded embedder identity "
                f"(embedder.txt is missing). It cannot be verified against the "
                f"configured embedder; rebuild the index (`rag index`) or restore "
                f"embedder.txt from the machine that built it."
            )
    else:
        stored = fingerprint_path.read_text().strip()
        if stored != embedder.fingerprint:
            raise IndexError_(
                f"index was built with embedder {stored} but config now specifies "
                f"{embedder.fingerprint}. Rebuild the index (`rag index`) or restore "
                f"the original embed config."
            )

    # The persisted kind must match the configured store before any store class
    # touches the files: the kinds share ids.json but keep distinct vector files,
    # so a config flip between builds would otherwise pair a stale vector file
    # with fresh ids and return wrong chunks with confident scores.
    meta_path = directory / "index_meta.json"
    if meta_path.exists():
        stored_kind = json.loads(meta_path.read_text()).get("kind")
        if stored_kind != config.index.store:
            raise IndexError_(
                f"index at {directory} was built with store {stored_kind!r} but "
                f"config.index.store is {config.index.store!r}. Rebuild the index "
                f"(`rag index`) or set index.store to {stored_kind!r}."
            )

    store = ChunkStore.load(directory)
    vectors: VectorStore
    if config.index.store == "faiss":
        vectors = FaissVectorStore.load(directory)
    else:
        vectors = InMemoryVectorStore.load(directory)

    lexical: LexicalIndex | None = None
    if config.index.bm25:
        # A missing or corrupt BM25 file under bm25=true is config drift, the same
        # class of fault as an embedder mismatch. Degrading to dense-only silently
        # would let an eval report attribute dense-only numbers to a hybrid config.
        try:
            lexical = Bm25Index.load(directory)
        except IndexError_ as exc:
            raise IndexError_(
                f"config.index.bm25 is true but no usable BM25 index was found: {exc} "
                f"Rebuild the index (`rag index`) or set index.bm25 to false."
            ) from exc

    return IndexBundle(store=store, vectors=vectors, lexical=lexical, embedder=embedder)


_SCOPE_SAMPLE_LIMIT = 512


def sample_scope_chunks(chunks: Sequence[Chunk], limit: int = _SCOPE_SAMPLE_LIMIT) -> list[Chunk]:
    """Evenly strided sample of the corpus for the scope-classifier centroid.

    Store order is manifest/ingest order, so a prefix of the store is the first
    fifteen or so papers, not a sample: a centroid fitted on it refuses legitimate
    questions about papers listed later in the manifest. The stride touches every
    region of the corpus and, unlike a random sample, is deterministic across
    processes, so two machines fit the same centroid from the same index.
    """
    if len(chunks) <= limit:
        return list(chunks)
    stride = len(chunks) / limit
    return [chunks[int(i * stride)] for i in range(limit)]


def build_pipeline(
    config: Config,
    bundle: IndexBundle,
    *,
    client: LlmClient | None = None,
    secrets: Secrets | None = None,
) -> Pipeline:
    """Assemble the full ask path."""
    secrets = secrets or Secrets()
    anthropic_key = (
        secrets.anthropic_api_key.get_secret_value() if secrets.anthropic_api_key else None
    )
    cohere_key = secrets.cohere_api_key.get_secret_value() if secrets.cohere_api_key else None
    openai_key = secrets.openai_api_key.get_secret_value() if secrets.openai_api_key else None

    client = client or build_client(
        config.generate.provider,
        model=config.generate.model,
        ollama_model=config.generate.ollama_model,
        ollama_host=config.generate.ollama_host,
        api_key=anthropic_key,
        openai_api_key=openai_key,
    )

    scope = ScopeClassifier(bundle.embedder)
    if len(bundle.store):
        # Fit the scope classifier on a stride across the whole corpus, at most 512
        # chunks: the centroid converges long before that and startup stays instant.
        chunks = sample_scope_chunks(bundle.store.chunks)
        scope.fit(np.asarray(bundle.embedder.embed_documents([c.text for c in chunks])))
        # The fit is the only post-build path that embeds corpus text. When the
        # index directory was shipped without its embed cache, those embeddings are
        # misses; persist them here or every process start pays the cost again.
        bundle.flush_embed_cache()

    transform = build_transform(
        config.retrieve.strategy,
        client,
        count=config.retrieve.multi_query_count,
        model=config.generate.model,
        effort="low",  # rewrites are mechanical; spend effort on the answer instead
    )
    reranker = build_reranker(config.retrieve, cohere_api_key=cohere_key)

    retriever = Retriever(
        store=bundle.store,
        vectors=bundle.vectors,
        embedder=bundle.embedder,
        config=config.retrieve,
        guardrails=config.guardrails,
        lexical=bundle.lexical,
        transform=transform,
        reranker=reranker,
    )

    return Pipeline(
        input_guard=InputGuard(config.guardrails, scope=scope),
        retriever=retriever,
        retrieval_guard=RetrievalGuard(config.guardrails),
        answerer=Answerer(client, OutputGuard(config.guardrails), config.generate),
    )
