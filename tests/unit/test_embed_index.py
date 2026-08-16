import numpy as np
import pytest

from rag.domain import Chunk, make_chunk_id
from rag.embed.models import CachedEmbedder, FakeEmbedder
from rag.errors import IndexError_
from rag.index.base import ChunkStore, Hit
from rag.index.stores import Bm25Index, InMemoryVectorStore, tokenize
from rag.retrieve.fusion import cap_per_document, deduplicate, reciprocal_rank_fusion


def _chunk(doc_id: str, text: str, start: int = 0) -> Chunk:
    end = start + len(text)
    return Chunk(
        chunk_id=make_chunk_id(doc_id, start, end, 0),
        doc_id=doc_id,
        doc_title=doc_id.title(),
        text=text,
        char_start=start,
        char_end=end,
        section_title="S",
        page_start=1,
        page_end=1,
    )


class TestFakeEmbedder:
    def test_deterministic(self):
        embedder = FakeEmbedder()
        a = embedder.embed_query("selective state spaces")
        b = embedder.embed_query("selective state spaces")
        np.testing.assert_array_equal(a, b)

    def test_similar_texts_are_closer(self):
        embedder = FakeEmbedder(dimension=64)
        base = embedder.embed_query("mamba selective state space model")
        near = embedder.embed_query("mamba selective state space architecture")
        far = embedder.embed_query("pizza recipes with extra cheese")
        assert float(base @ near) > float(base @ far)

    def test_normalised(self):
        embedder = FakeEmbedder()
        assert abs(float(np.linalg.norm(embedder.embed_query("hello world"))) - 1.0) < 1e-5


class TestCachedEmbedder:
    def test_cache_avoids_recompute(self, tmp_path):
        inner = FakeEmbedder()
        cached = CachedEmbedder(inner, tmp_path)
        cached.embed_documents(["a b c", "d e f"])
        calls_after_first = inner.call_count
        cached.embed_documents(["a b c", "d e f"])
        assert inner.call_count == calls_after_first
        assert cached.hits == 2

    def test_cache_persists_across_instances(self, tmp_path):
        first = CachedEmbedder(FakeEmbedder(), tmp_path)
        expected = first.embed_documents(["persistent text"])
        first.flush()

        inner = FakeEmbedder()
        second = CachedEmbedder(inner, tmp_path)
        result = second.embed_documents(["persistent text"])
        np.testing.assert_array_almost_equal(result, expected)
        assert inner.call_count == 0


class TestInMemoryVectorStore:
    def test_ranks_by_cosine(self):
        embedder = FakeEmbedder(dimension=64)
        store = InMemoryVectorStore(dimension=64)
        texts = {
            "a": "mamba selective state space model scan",
            "b": "vision transformer image patches",
            "c": "mamba hardware aware scan implementation",
        }
        store.add(list(texts), embedder.embed_documents(list(texts.values())))
        hits = store.search(embedder.embed_query("mamba state space scan"), k=3)
        assert hits[0].chunk_id in {"a", "c"}
        assert hits[-1].chunk_id == "b"

    def test_save_load_round_trip(self, tmp_path):
        embedder = FakeEmbedder(dimension=16)
        store = InMemoryVectorStore(dimension=16)
        store.add(["x", "y"], embedder.embed_documents(["first text", "second text"]))
        store.save(tmp_path)
        loaded = InMemoryVectorStore.load(tmp_path)
        query = embedder.embed_query("first text")
        assert [h.chunk_id for h in loaded.search(query, 2)] == [
            h.chunk_id for h in store.search(query, 2)
        ]

    def test_dimension_mismatch_raises(self):
        store = InMemoryVectorStore(dimension=8)
        with pytest.raises(IndexError_):
            store.add(["x"], np.zeros((1, 16), dtype=np.float32))

    def test_empty_search(self):
        assert InMemoryVectorStore(dimension=8).search(np.zeros(8, dtype=np.float32), 5) == []


class TestBm25:
    def test_exact_term_wins(self):
        index = Bm25Index()
        index.add(
            ["flash", "generic"],
            [
                "FlashAttention tiles the computation to reduce HBM accesses",
                "attention mechanisms are widely used in sequence models",
            ],
        )
        hits = index.search("flashattention HBM", k=2)
        assert hits and hits[0].chunk_id == "flash"

    def test_hyphenated_tokens_survive(self):
        assert "gpt-4" in tokenize("Comparing GPT-4 and bge-m3 embeddings v1.5")
        assert "bge-m3" in tokenize("Comparing GPT-4 and bge-m3 embeddings v1.5")

    def test_no_match_returns_empty(self):
        index = Bm25Index()
        index.add(["a"], ["completely unrelated content"])
        assert index.search("zzzz qqqq", k=5) == []

    def test_save_load(self, tmp_path):
        index = Bm25Index()
        index.add(["a", "b"], ["mamba scan text", "vision patches text"])
        index.save(tmp_path)
        loaded = Bm25Index.load(tmp_path)
        assert loaded.search("mamba", k=1)[0].chunk_id == "a"


class TestChunkStore:
    def test_round_trip(self, tmp_path):
        chunks = [_chunk("mamba", "text one"), _chunk("bert", "text two")]
        store = ChunkStore(chunks)
        store.save(tmp_path)
        loaded = ChunkStore.load(tmp_path)
        assert len(loaded) == 2
        assert loaded.get(chunks[0].chunk_id).text == "text one"

    def test_resolve_skips_unknown(self):
        chunk = _chunk("mamba", "text")
        store = ChunkStore([chunk])
        resolved = store.resolve([Hit(chunk.chunk_id, 0.9), Hit("missing", 0.8)])
        assert len(resolved) == 1

    def test_get_unknown_raises(self):
        with pytest.raises(IndexError_, match="rebuild"):
            ChunkStore([]).get("nope")


class TestFusion:
    def test_rrf_rewards_agreement(self):
        dense = [Hit("a", 0.9), Hit("b", 0.8), Hit("c", 0.7)]
        lexical = [Hit("b", 12.0), Hit("a", 9.0), Hit("d", 3.0)]
        fused = reciprocal_rank_fusion([dense, lexical])
        top_two = {fused[0].chunk_id, fused[1].chunk_id}
        assert top_two == {"a", "b"}  # both appear in both lists
        assert fused[0].score > fused[2].score

    def test_rrf_deterministic_tiebreak(self):
        first = reciprocal_rank_fusion([[Hit("a", 1.0)], [Hit("b", 1.0)]])
        second = reciprocal_rank_fusion([[Hit("a", 1.0)], [Hit("b", 1.0)]])
        assert [h.chunk_id for h in first] == [h.chunk_id for h in second]

    def test_dedup_drops_near_identical(self):
        text = "the selective scan mechanism makes parameters input dependent " * 4
        a = (_chunk("mamba", text, 0), 0.9)
        b = (_chunk("mamba", text + " extra", 1000), 0.8)
        c = (_chunk("bert", "completely different content about bidirectional encoders", 0), 0.7)
        kept = deduplicate([a, b, c], threshold=0.6)
        assert len(kept) == 2
        assert kept[0][0].doc_id == "mamba"
        assert kept[1][0].doc_id == "bert"

    def test_cap_per_document(self):
        pairs = [(_chunk("mamba", f"text {i}", i * 100), 1.0 - i * 0.1) for i in range(4)] + [
            (_chunk("bert", "bert text", 0), 0.5)
        ]
        capped = cap_per_document(pairs, max_per_doc=2)
        assert sum(1 for c, _ in capped if c.doc_id == "mamba") == 2
        assert sum(1 for c, _ in capped if c.doc_id == "bert") == 1
