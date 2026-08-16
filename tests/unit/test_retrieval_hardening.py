"""Regression tests for the retrieval hardening review findings.

Each test here pins a failure mode that shipped once: per-doc capping starving
top_k, dedup that never fired, indexes that loaded under the wrong embedder or
store kind, a relevance floor measured against invented HyDE text, and caches
that silently never persisted. If one of these breaks again, the matching test
names the original failure.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from rag.app import (
    build_index,
    build_ocr_engine,
    build_pipeline,
    load_index,
    sample_scope_chunks,
    save_index,
)
from rag.config import Config, GuardrailConfig, RetrieveConfig
from rag.domain import Action, Chunk, make_chunk_id
from rag.embed.models import CachedEmbedder, FakeEmbedder
from rag.errors import IndexError_
from rag.generate.client import FakeLlmClient
from rag.guardrails.retrieval_guard import RetrievalGuard
from rag.index.base import ChunkStore, Hit
from rag.index.stores import Bm25Index, InMemoryVectorStore
from rag.ingest.ocr.cached import save_ocr_document
from rag.ingest.ocr.fake import FakeOcrEngine, build_document
from rag.retrieve.fusion import deduplicate, reciprocal_rank_fusion
from rag.retrieve.rerank import NoopReranker
from rag.retrieve.retriever import Retriever
from rag.retrieve.rewrite import HydeTransform, IdentityTransform, RewriteResult
from tests.conftest import PAPER_MARKUP, make_chunks

try:
    import faiss  # noqa: F401

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _chunk(text: str, doc_id: str, start: int = 0) -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id(doc_id, start, start + len(text), 0),
        doc_id=doc_id,
        doc_title=doc_id.title(),
        text=text,
        char_start=start,
        char_end=start + len(text),
        section_title="Method",
        section_number="3",
        page_start=1,
        page_end=1,
    )


def _make_config(data_dir: Path, **overrides: object) -> Config:
    payload: dict[str, object] = {
        "paths": {"data": str(data_dir)},
        "ocr": {"engine": "fake"},
        "chunk": {"max_chunk_tokens": 128, "part_overlap_tokens": 16, "min_chunk_chars": 80},
        "embed": {"provider": "fake", "model": "fake", "dimension": 32},
        "index": {"store": "inmemory", "bm25": True},
        "retrieve": {"strategy": "vanilla", "top_k": 4, "fetch_k": 10, "rerank": False},
        "generate": {"provider": "fake"},
        "guardrails": {"relevance_floor": 0.05, "scope_threshold": -1.0},
    }
    payload.update(overrides)
    return Config.model_validate(payload)


class _PresetVectors:
    """VectorStore stand-in that returns one fixed ranking for every query."""

    def __init__(self, hits: list[Hit]) -> None:
        self._hits = list(hits)

    @property
    def size(self) -> int:
        return len(self._hits)

    def search(self, query: object, k: int) -> list[Hit]:
        return self._hits[:k]


class _PresetLexical:
    """LexicalIndex stand-in that returns one fixed ranking for every query."""

    def __init__(self, hits: list[Hit]) -> None:
        self._hits = list(hits)

    @property
    def size(self) -> int:
        return len(self._hits)

    def search(self, query: str, k: int) -> list[Hit]:
        return self._hits[:k]


def _retriever(**overrides: object) -> Retriever:
    defaults: dict[str, object] = {
        "config": RetrieveConfig(
            strategy="vanilla", top_k=4, fetch_k=20, rerank=False, max_per_doc=2
        ),
        "guardrails": GuardrailConfig(),
        "embedder": FakeEmbedder(),
        "lexical": None,
        "transform": IdentityTransform(),
        "reranker": NoopReranker(),
    }
    defaults.update(overrides)
    return Retriever(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Per-document cap must not starve top_k (finding: cap after a narrow window)
# --------------------------------------------------------------------------- #


class TestCapDoesNotStarveTopK:
    def test_dominating_document_cannot_shrink_the_result_set(self) -> None:
        # One paper owns the head of the ranking; cap-valid chunks from two other
        # papers sit past where a top_k*2 rerank window used to end.
        chunks: list[Chunk] = []
        hits: list[Hit] = []
        for i in range(8):
            chunk = _chunk(f"doc a chunk number {i} text", "doc-a", start=i * 100)
            chunks.append(chunk)
            hits.append(Hit(chunk_id=chunk.chunk_id, score=0.9 - i * 0.01))
        for doc in ("doc-b", "doc-c"):
            for i in range(3):
                chunk = _chunk(f"{doc} body chunk {i} words", doc, start=i * 100)
                chunks.append(chunk)
                hits.append(Hit(chunk_id=chunk.chunk_id, score=0.5 - i * 0.01))

        retriever = _retriever(store=ChunkStore(chunks), vectors=_PresetVectors(hits))
        retrieval = retriever.retrieve("a query dominated by one paper")

        assert len(retrieval.results) == 4  # full top_k, not 2
        per_doc = Counter(s.chunk.doc_id for s in retrieval.results)
        assert all(count <= 2 for count in per_doc.values())
        assert len(per_doc) >= 2  # refilled from the other papers


# --------------------------------------------------------------------------- #
# Dedup uses containment (finding: 0.97 Jaccard on shingles never fired)
# --------------------------------------------------------------------------- #


class TestDeduplicateContainment:
    THRESHOLD = GuardrailConfig().dedup_threshold  # what the retriever passes

    def test_exact_duplicate_under_different_chunk_id_dropped(self) -> None:
        text = " ".join(f"alpha{i}" for i in range(60))
        original = _chunk(text, "doc-a")
        copy = _chunk(text, "doc-b")  # content-addressing gives it a new id
        kept = deduplicate([(original, 0.9), (copy, 0.8)], threshold=self.THRESHOLD)
        assert [c.chunk_id for c, _ in kept] == [original.chunk_id]

    def test_fully_contained_chunk_dropped(self) -> None:
        big_text = " ".join(f"alpha{i}" for i in range(60))
        quoted = " ".join(f"alpha{i}" for i in range(20, 45))  # verbatim slice
        big = _chunk(big_text, "doc-a")
        small = _chunk(quoted, "doc-b")
        kept = deduplicate([(big, 0.9), (small, 0.8)], threshold=self.THRESHOLD)
        assert [c.chunk_id for c, _ in kept] == [big.chunk_id]

    def test_adjacent_split_parts_are_retained(self) -> None:
        # Split parts share only their construction overlap; the rest of each part
        # is unique content and must survive dedup.
        shared = " ".join(f"shared{i}" for i in range(20))
        part_one = " ".join(f"head{i}" for i in range(80)) + " " + shared
        part_two = shared + " " + " ".join(f"tail{i}" for i in range(80))
        first = _chunk(part_one, "doc-a", start=0)
        second = _chunk(part_two, "doc-a", start=400)
        kept = deduplicate([(first, 0.9), (second, 0.8)], threshold=self.THRESHOLD)
        assert len(kept) == 2


# --------------------------------------------------------------------------- #
# Index identity checks (findings: fail-open embedder.txt, unchecked store kind,
# silently degrading BM25)
# --------------------------------------------------------------------------- #


class TestIndexIdentity:
    def test_missing_embedder_fingerprint_is_a_hard_error(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path / "data")
        bundle = build_index(config, list(make_chunks(PAPER_MARKUP)))
        save_index(bundle, config.paths.index)

        (config.paths.index / "embedder.txt").unlink()
        with pytest.raises(IndexError_, match="embedder"):
            load_index(config)

    def test_kind_mismatch_rejected_before_store_load(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path / "data", index={"store": "inmemory", "bm25": True})
        bundle = build_index(config, list(make_chunks(PAPER_MARKUP)))
        save_index(bundle, config.paths.index)

        flipped = _make_config(tmp_path / "data", index={"store": "faiss", "bm25": True})
        with pytest.raises(IndexError_, match="was built with store"):
            load_index(flipped)

    def test_store_load_refuses_other_kinds_meta(self, tmp_path: Path) -> None:
        store = InMemoryVectorStore(dimension=4)
        store.add(["only"], np.ones((1, 4), dtype=np.float32))
        store.save(tmp_path)
        (tmp_path / "index_meta.json").write_text(json.dumps({"kind": "faiss", "dimension": 4}))

        with pytest.raises(IndexError_, match="built as 'faiss'"):
            InMemoryVectorStore.load(tmp_path)

    def test_missing_bm25_under_bm25_config_is_a_hard_error(self, tmp_path: Path) -> None:
        built = _make_config(tmp_path / "data", index={"store": "inmemory", "bm25": False})
        bundle = build_index(built, list(make_chunks(PAPER_MARKUP)))
        save_index(bundle, built.paths.index)

        loading = _make_config(tmp_path / "data", index={"store": "inmemory", "bm25": True})
        with pytest.raises(IndexError_, match="BM25"):
            load_index(loading)

    def test_corrupt_bm25_file_is_a_typed_error(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path / "data")
        bundle = build_index(config, list(make_chunks(PAPER_MARKUP)))
        save_index(bundle, config.paths.index)

        (config.paths.index / Bm25Index.FILENAME).write_text("{not valid json")
        with pytest.raises(IndexError_, match="corrupt BM25"):
            load_index(config)


# --------------------------------------------------------------------------- #
# Relevance floor score space (findings: fallback to fused scores, HyDE text
# setting the floor signal, lexical-only retrieval refused on the floor)
# --------------------------------------------------------------------------- #


class TestRelevanceFloorSignal:
    def test_dense_score_is_a_required_argument(self) -> None:
        from rag.domain import Scored

        chunk = _chunk("some retrieved text here", "doc-a")
        scored = (Scored(chunk=chunk, score=0.9, rank=1, retriever="test"),)
        with pytest.raises(TypeError):
            RetrievalGuard(GuardrailConfig()).check(scored)  # type: ignore[call-arg]

    def test_floor_thresholds_dense_score_not_fused_score(self) -> None:
        from rag.domain import Scored

        chunk = _chunk("some retrieved text here", "doc-a")
        guard = RetrievalGuard(GuardrailConfig(relevance_floor=0.25))

        # High fused score cannot rescue a below-floor dense score.
        high_fused = (Scored(chunk=chunk, score=0.9, rank=1, retriever="test"),)
        assert not guard.check(high_fused, top_dense_score=0.05).allowed

        # RRF-scale fused scores (~0.016) must not trip the floor when the dense
        # cosine is healthy.
        rrf_scale = (Scored(chunk=chunk, score=0.016, rank=1, retriever="test"),)
        assert guard.check(rrf_scale, top_dense_score=0.9).allowed

    def test_hyde_hypothetical_does_not_set_the_floor_signal(self) -> None:
        texts = [
            "the gating network scores attention heads and skips weak ones entirely",
            "selective pruning of transformer heads preserves accuracy at less compute",
            "a sparsity penalty trains the gate jointly with the base model weights",
        ]
        chunks = [_chunk(text, f"doc-{i}", start=i * 10) for i, text in enumerate(texts)]
        embedder = FakeEmbedder(dimension=64)
        vectors = InMemoryVectorStore(dimension=64)
        vectors.add([c.chunk_id for c in chunks], embedder.embed_documents(texts))

        original = "completely unrelated question about zebra migration patterns"
        hypothetical = chunks[0].text  # embeds identically to a corpus chunk

        original_top = max(h.score for h in vectors.search(embedder.embed_query(original), 10))
        hyde_top = max(h.score for h in vectors.search(embedder.embed_query(hypothetical), 10))
        assert hyde_top > original_top  # otherwise this test proves nothing

        retriever = _retriever(
            store=ChunkStore(chunks),
            vectors=vectors,
            embedder=embedder,
            config=RetrieveConfig(strategy="hyde", top_k=4, fetch_k=10, rerank=False),
            transform=HydeTransform(FakeLlmClient([hypothetical])),
        )
        retrieval = retriever.retrieve(original)

        assert retrieval.top_dense_score == pytest.approx(original_top)
        assert retrieval.top_dense_score is not None
        assert retrieval.top_dense_score < hyde_top

    def test_transform_dropping_the_original_still_fails_closed(self) -> None:
        # A future transform that omits the user's query from its rewrites must not
        # silently hand the floor to rewritten text: the retriever runs one extra
        # dense search on the original.
        class _DropsOriginal:
            @property
            def name(self) -> str:
                return "drops-original"

            def transform(self, query: str) -> RewriteResult:
                return RewriteResult(queries=("selective attention heads gating",), strategy="x")

        texts = ["selective attention heads gating text body", "another unrelated chunk of prose"]
        chunks = [_chunk(text, f"doc-{i}", start=i * 10) for i, text in enumerate(texts)]
        embedder = FakeEmbedder(dimension=64)
        vectors = InMemoryVectorStore(dimension=64)
        vectors.add([c.chunk_id for c in chunks], embedder.embed_documents(texts))

        original = "question about zebra migration"
        original_top = max(h.score for h in vectors.search(embedder.embed_query(original), 10))

        retriever = _retriever(
            store=ChunkStore(chunks),
            vectors=vectors,
            embedder=embedder,
            transform=_DropsOriginal(),
        )
        retrieval = retriever.retrieve(original)
        assert retrieval.top_dense_score == pytest.approx(original_top)

    def test_lexical_only_results_survive_the_floor(self) -> None:
        # Empty vector store, intact BM25: exact-term hits are sufficient evidence
        # and must be allowed with a note, not refused against a floor that has no
        # cosine to measure.
        texts = [
            "flashattention-2 tiles the attention computation across warps",
            "an unrelated section about optimizer schedules and warmup",
        ]
        chunks = [_chunk(text, f"doc-{i}", start=i * 10) for i, text in enumerate(texts)]
        lexical = Bm25Index()
        lexical.add([c.chunk_id for c in chunks], texts)

        retriever = _retriever(
            store=ChunkStore(chunks),
            vectors=InMemoryVectorStore(dimension=32),  # size 0: dense is broken
            lexical=lexical,
        )
        retrieval = retriever.retrieve("flashattention-2 kernel details")

        assert retrieval.results
        assert retrieval.top_dense_score is None

        verdict = RetrievalGuard(GuardrailConfig()).check(
            retrieval.results, top_dense_score=retrieval.top_dense_score
        )
        assert verdict.allowed
        floor = [d for d in verdict.decisions if d.rule_id == "retrieval.relevance_floor"]
        assert floor and floor[0].action is Action.ALLOW
        assert "no dense signal" in floor[0].reason


# --------------------------------------------------------------------------- #
# Scope classifier sampling (finding: fitted on a document-ordered prefix)
# --------------------------------------------------------------------------- #


class TestScopeSampling:
    def test_small_corpus_used_whole(self) -> None:
        chunks = [_chunk(f"text {i}", "doc-a", start=i * 10) for i in range(20)]
        assert sample_scope_chunks(chunks) == chunks

    def test_stride_covers_documents_past_the_limit(self) -> None:
        # Manifest order: 600 NLP chunks, then 600 RL chunks. A prefix of 512
        # would contain zero RL chunks and the centroid would refuse RL questions.
        early = [_chunk(f"grammar parsing tokens {i}", "nlp", start=i * 10) for i in range(600)]
        late = [_chunk(f"replay buffer reward {i}", "rl", start=i * 10) for i in range(600)]
        sample = sample_scope_chunks(early + late)

        assert len(sample) == 512
        per_doc = Counter(c.doc_id for c in sample)
        assert per_doc["rl"] > 0
        # Even coverage: each half of the corpus contributes about half the sample.
        assert abs(per_doc["nlp"] - per_doc["rl"]) <= 2

    def test_stride_is_deterministic(self) -> None:
        chunks = [_chunk(f"text {i}", "doc-a", start=i * 10) for i in range(1000)]
        first = [c.chunk_id for c in sample_scope_chunks(chunks)]
        second = [c.chunk_id for c in sample_scope_chunks(chunks)]
        assert first == second


# --------------------------------------------------------------------------- #
# Embed cache persistence (finding: flush only ever happened in save_index)
# --------------------------------------------------------------------------- #


class TestEmbedCacheFlush:
    def test_build_pipeline_persists_scope_fit_embeddings(self, config: Config) -> None:
        chunks = list(make_chunks(PAPER_MARKUP, doc_id="selective"))
        bundle = build_index(config, chunks)

        cache_root = config.paths.index / "embed-cache"
        store_file = cache_root / bundle.embedder.fingerprint / "vectors.npz"
        assert not store_file.exists()  # build alone keeps the cache in memory

        build_pipeline(config, bundle, client=FakeLlmClient([]))
        assert store_file.exists()

        # A fresh process over the same cache directory re-embeds nothing.
        fresh = CachedEmbedder(FakeEmbedder(), cache_root)
        fresh.embed_documents([c.text for c in chunks])
        assert fresh.misses == 0
        assert fresh.hits == len(chunks)


# --------------------------------------------------------------------------- #
# Store determinism and backend parity (findings: unstable tie ordering, the
# documented FAISS parity test not existing)
# --------------------------------------------------------------------------- #


class TestStoreOrdering:
    def test_inmemory_tied_scores_keep_insertion_order(self) -> None:
        ids = ["zeta", "alpha", "mid", "beta", "last"]
        vectors = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (5, 1))
        store = InMemoryVectorStore(dimension=4)
        store.add(ids, vectors)
        hits = store.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 5)
        assert [h.chunk_id for h in hits] == ids

    def test_bm25_tied_scores_keep_insertion_order(self) -> None:
        ids = ["zeta", "alpha", "mid"]
        index = Bm25Index()
        index.add(ids, ["the same exact text here"] * 3)
        hits = index.search("exact text", 3)
        assert [h.chunk_id for h in hits] == ids

    @pytest.mark.skipif(not HAS_FAISS, reason="faiss-cpu is not installed")
    def test_faiss_matches_inmemory_reference(self) -> None:
        from rag.index.stores import FaissVectorStore

        rng = np.random.default_rng(7)
        vectors = rng.normal(size=(50, 16)).astype(np.float32)  # tie-free
        ids = [f"chunk-{i:03d}" for i in range(50)]

        reference = InMemoryVectorStore(dimension=16)
        reference.add(ids, vectors)
        fast = FaissVectorStore(dimension=16)
        fast.add(ids, vectors)

        for query in rng.normal(size=(5, 16)).astype(np.float32):
            expected = reference.search(query, 10)
            actual = fast.search(query, 10)
            assert [h.chunk_id for h in actual] == [h.chunk_id for h in expected]
            for got, want in zip(actual, expected, strict=True):
                assert got.score == pytest.approx(want.score, abs=1e-5)


# --------------------------------------------------------------------------- #
# Fusion weights (finding: dead weights parameter, multi-query drowning BM25)
# --------------------------------------------------------------------------- #


class TestFusionWeights:
    def test_weights_balance_dense_block_against_lexical(self) -> None:
        dense = [Hit(chunk_id="a", score=0.9), Hit(chunk_id="b", score=0.8)]
        lexical = [Hit(chunk_id="b", score=7.0), Hit(chunk_id="a", score=5.0)]

        # Two dense rankings at half weight carry the same total vote as one, so
        # dense and lexical stay balanced regardless of the rewrite strategy.
        fused = reciprocal_rank_fusion([dense, dense, lexical], weights=[0.5, 0.5, 1.0])
        scores = {h.chunk_id: h.score for h in fused}
        assert scores["a"] == pytest.approx(scores["b"])

        # Unweighted, the duplicated dense list double-votes and drowns lexical.
        drowned = {h.chunk_id: h.score for h in reciprocal_rank_fusion([dense, dense, lexical])}
        assert drowned["a"] > drowned["b"]

    def test_weights_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="weights"):
            reciprocal_rank_fusion([[Hit(chunk_id="a", score=1.0)]], weights=[1.0, 2.0])

    def test_retriever_normalizes_multi_query_dense_weight(self) -> None:
        class _TwoQueries:
            @property
            def name(self) -> str:
                return "two"

            def transform(self, query: str) -> RewriteResult:
                return RewriteResult(queries=(query, query + " rephrased"), strategy="multi")

        chunk_a = _chunk("dense favourite chunk text", "doc-a")
        chunk_b = _chunk("lexical favourite chunk words", "doc-b")
        retriever = _retriever(
            store=ChunkStore([chunk_a, chunk_b]),
            vectors=_PresetVectors([Hit(chunk_id=chunk_a.chunk_id, score=0.9)]),
            lexical=_PresetLexical([Hit(chunk_id=chunk_b.chunk_id, score=6.0)]),
            transform=_TwoQueries(),
        )
        retrieval = retriever.retrieve("a query")

        # Two dense rankings for chunk A fuse to the same score as one lexical
        # ranking for chunk B: the dense block carries constant total weight.
        assert len(retrieval.results) == 2
        assert retrieval.results[0].score == pytest.approx(retrieval.results[1].score)


# --------------------------------------------------------------------------- #
# Fake OCR engine wiring (finding: fixtures dir doubled as the cache write dir)
# --------------------------------------------------------------------------- #


class TestFakeOcrWiring:
    def test_fake_engine_is_unwrapped_and_writes_nothing(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path / "data", ocr={"engine": "fake"})
        config.paths.ocr_cache.mkdir(parents=True, exist_ok=True)
        document = build_document(
            "paper1", "[title] A Title\n\nBody text long enough to form one block."
        )
        save_ocr_document(document, config.paths.ocr_cache / "paper1.json")

        engine = build_ocr_engine(config)
        assert isinstance(engine, FakeOcrEngine)  # not cache-wrapped

        before = sorted(p.name for p in config.paths.ocr_cache.iterdir())
        # The fake never opens the PDF, so a missing path must be fine and no
        # cache-keyed duplicate may be written back into the fixtures directory.
        result = engine.read(Path("does-not-exist.pdf"), "paper1")
        after = sorted(p.name for p in config.paths.ocr_cache.iterdir())

        assert result.doc_id == "paper1"
        assert after == before
