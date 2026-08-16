import pytest
from pydantic import ValidationError

from rag.config import ChunkConfig, Config


class TestConfigHash:
    def test_stable_across_instances(self):
        assert Config().hash() == Config().hash()

    def test_changes_with_settings(self):
        base = Config()
        changed = Config.model_validate({"retrieve": {"top_k": 9}})
        assert base.hash() != changed.hash()

    def test_paths_do_not_affect_hash(self):
        base = Config()
        moved = Config.model_validate({"paths": {"data": "/somewhere/else"}})
        assert base.hash() == moved.hash()

    def test_ingest_hash_ignores_retrieval_changes(self):
        base = Config()
        retrieval_changed = Config.model_validate({"retrieve": {"top_k": 9}})
        chunk_changed = Config.model_validate({"chunk": {"max_chunk_tokens": 256}})
        assert base.ingest_hash() == retrieval_changed.ingest_hash()
        assert base.ingest_hash() != chunk_changed.ingest_hash()


class TestChunkConfig:
    def test_overlap_must_be_smaller_than_cap(self):
        with pytest.raises(ValidationError):
            ChunkConfig(max_chunk_tokens=128, part_overlap_tokens=128)

    def test_char_budgets(self):
        config = ChunkConfig(max_chunk_tokens=100, chars_per_token=4.0)
        assert config.max_chunk_chars == 400


class TestLoad:
    def test_defaults_without_file(self):
        config = Config.load(None)
        assert config.chunk.max_depth == 1

    def test_load_yaml(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("retrieve:\n  top_k: 7\n")
        assert Config.load(path).retrieve.top_k == 7

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValidationError):
            Config.model_validate({"retrieve": {"strategy": "quantum"}})
