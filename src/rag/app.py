"""The composition root.

The only module that knows which concrete implementation fills each protocol.
Everything else takes its dependencies as constructor arguments, and the tests
compose their own graph out of fakes. If a module other than this one constructs a
Surya engine or an Anthropic client, that is a review comment.
"""

from __future__ import annotations

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
    """OCR engine per config, always cache-wrapped."""
    from rag.ingest.ocr.cached import CachedOcrEngine

    inner: OcrEngine
    if config.ocr.engine == "fake":
        from rag.ingest.ocr.fake import FakeOcrEngine

        inner = FakeOcrEngine.from_fixtures(config.paths.ocr_cache)
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
    (directory / "embedder.txt").write_text(bundle.embedder.fingerprint)
    from rag.embed.models import CachedEmbedder

    if isinstance(bundle.embedder, CachedEmbedder):
        bundle.embedder.flush()


def load_index(config: Config, *, embedder: Embedder | None = None) -> IndexBundle:
    directory = config.paths.index
    embedder = embedder or build_embedder(config.embed, cache_dir=directory / "embed-cache")

    fingerprint_path = directory / "embedder.txt"
    if fingerprint_path.exists():
        stored = fingerprint_path.read_text().strip()
        if stored != embedder.fingerprint:
            raise IndexError_(
                f"index was built with embedder {stored} but config now specifies "
                f"{embedder.fingerprint}. Rebuild the index (`rag index`) or restore "
                f"the original embed config."
            )

    store = ChunkStore.load(directory)
    vectors: VectorStore
    if config.index.store == "faiss":
        vectors = FaissVectorStore.load(directory)
    else:
        vectors = InMemoryVectorStore.load(directory)

    lexical: LexicalIndex | None = None
    if config.index.bm25:
        try:
            lexical = Bm25Index.load(directory)
        except IndexError_:
            log.warning("BM25 index missing; continuing dense-only")

    return IndexBundle(store=store, vectors=vectors, lexical=lexical, embedder=embedder)


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

    client = client or build_client(
        config.generate.provider,
        model=config.generate.model,
        ollama_model=config.generate.ollama_model,
        ollama_host=config.generate.ollama_host,
        api_key=anthropic_key,
    )

    scope = ScopeClassifier(bundle.embedder)
    if len(bundle.store):
        # Fit the scope classifier on the actual corpus. Sampled at most 512 chunks:
        # the centroid converges long before that and this keeps startup instant.
        chunks = bundle.store.chunks[:512]
        scope.fit(np.asarray(bundle.embedder.embed_documents([c.text for c in chunks])))

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
