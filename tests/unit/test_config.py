import pytest
from pydantic import ValidationError

from rag.config import ChunkConfig, Config
from rag.errors import ConfigError


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

    def test_ingest_hash_ignores_ocr_throughput_and_caching(self):
        # Regression: flipping ocr.cache or ocr.batch_size changed ingest_hash
        # even though neither affects chunk content, so tooling read a spurious
        # "chunk-affecting config changed" signal.
        base = Config()
        cache_off = Config.model_validate({"ocr": {"cache": False}})
        batched = Config.model_validate({"ocr": {"batch_size": 8}})
        dpi_changed = Config.model_validate({"ocr": {"dpi": 200}})
        assert base.ingest_hash() == cache_off.ingest_hash()
        assert base.ingest_hash() == batched.ingest_hash()
        assert base.ingest_hash() != dpi_changed.ingest_hash()  # dpi does affect output


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

    def test_missing_file_raises_config_error(self, tmp_path):
        # Regression: a typo'd --config path surfaced as a bare
        # FileNotFoundError traceback instead of the package's typed error.
        missing = tmp_path / "nope.yaml"
        with pytest.raises(ConfigError, match=r"nope\.yaml"):
            Config.load(missing)

    def test_invalid_yaml_raises_config_error(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("retrieve: [unclosed\n")
        with pytest.raises(ConfigError, match="not valid YAML"):
            Config.load(path)
