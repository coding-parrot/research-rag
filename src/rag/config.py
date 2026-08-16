"""Typed configuration.

One config object drives every stage. It is hashable, so a run manifest can record
exactly what produced a set of eval numbers. Secrets come from the environment and
are never written to a config file or a log line.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag.errors import ConfigError

RepoRoot = Path(__file__).resolve().parents[2]


class PathsConfig(BaseModel):
    """All filesystem locations, resolved relative to the repo root."""

    data: Path = RepoRoot / "data"
    corpus_manifest: Path = RepoRoot / "corpus" / "corpus.yaml"
    evals: Path = RepoRoot / "evals"

    @property
    def pdfs(self) -> Path:
        return self.data / "pdfs"

    @property
    def ocr_cache(self) -> Path:
        return self.data / "ocr"

    @property
    def index(self) -> Path:
        return self.data / "index"

    @property
    def runs(self) -> Path:
        return self.data / "runs"

    def ensure(self) -> None:
        for p in (self.data, self.pdfs, self.ocr_cache, self.index, self.runs):
            p.mkdir(parents=True, exist_ok=True)


class OcrConfig(BaseModel):
    engine: Literal["surya", "pypdfium", "fake"] = "pypdfium"
    dpi: int = Field(default=150, ge=72, le=400)
    # Surya is slow enough that batching matters even on MPS.
    batch_size: int = Field(default=4, ge=1, le=64)
    # Layout regions below this confidence are dropped as noise.
    min_block_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    detect_tables: bool = True
    cache: bool = True


class HeaderConfig(BaseModel):
    """Header detection is the critical path: with one chunker there is no fallback."""

    use_outline: bool = True
    use_layout: bool = True
    use_regex: bool = True
    use_font: bool = False  # last-resort signal, off by default (noisy on arXiv)

    # A heading must clear this to be treated as a real section boundary.
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Two detections within this many characters are the same heading.
    merge_window: int = Field(default=120, ge=0)
    # Longest plausible heading. Anything longer is a paragraph that starts with a number.
    max_title_chars: int = Field(default=90, ge=10)
    # Top-level section numbers above this are almost certainly a list, not a section.
    max_section_number: int = Field(default=20, ge=1)


class ChunkConfig(BaseModel):
    """Section chunking, plus the two policies that make it survive a real paper."""

    # 1 == "3. Experiments" is one chunk and 3.1/3.2 stay inside it.
    max_depth: int = Field(default=1, ge=1, le=3)
    # Sections longer than this split into parts, each inheriting the section header.
    max_chunk_tokens: int = Field(default=512, ge=64)
    # Overlap between parts of a split section, in tokens.
    part_overlap_tokens: int = Field(default=64, ge=0)
    # Sections shorter than this merge forward into the next section.
    min_chunk_chars: int = Field(default=200, ge=0)
    # Rough chars-per-token used for budgeting. Deliberately conservative.
    chars_per_token: float = Field(default=4.0, gt=0)
    # Prepend the section header to every part so a mid-section part is self-describing.
    repeat_header_in_parts: bool = True
    # Text before the first detected heading (title block, abstract).
    frontmatter_section_title: str = "Abstract and frontmatter"

    @property
    def max_chunk_chars(self) -> int:
        return int(self.max_chunk_tokens * self.chars_per_token)

    @property
    def part_overlap_chars(self) -> int:
        return int(self.part_overlap_tokens * self.chars_per_token)

    @field_validator("part_overlap_tokens")
    @classmethod
    def _overlap_fits(cls, v: int, info: Any) -> int:
        cap = info.data.get("max_chunk_tokens", 512)
        if v >= cap:
            raise ValueError("part_overlap_tokens must be smaller than max_chunk_tokens")
        return v


class EmbedConfig(BaseModel):
    provider: Literal["sentence-transformers", "fake"] = "sentence-transformers"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = Field(default=32, ge=1)
    normalize: bool = True
    cache: bool = True


class IndexConfig(BaseModel):
    store: Literal["faiss", "inmemory"] = "faiss"
    # Lexical half of hybrid retrieval. BM25 catches exact terms embeddings blur.
    bm25: bool = True


class RetrieveConfig(BaseModel):
    strategy: Literal["vanilla", "multi_query", "hyde", "hybrid"] = "hybrid"
    top_k: int = Field(default=4, ge=1, le=50)
    # Candidate pool pulled before reranking.
    fetch_k: int = Field(default=20, ge=1, le=200)
    # Off by default: the cross-encoder is a 1.1GB download. Flip on once cached.
    rerank: bool = False
    reranker: Literal["cross-encoder", "cohere", "none"] = "cross-encoder"
    reranker_model: str = "BAAI/bge-reranker-base"
    cohere_rerank_model: str = "rerank-v4.0-fast"
    # Reciprocal rank fusion constant for hybrid retrieval.
    rrf_k: int = Field(default=60, ge=1)
    multi_query_count: int = Field(default=3, ge=1, le=8)
    # At most this many chunks from any single paper, so one paper cannot own the prompt.
    max_per_doc: int = Field(default=2, ge=1)


class GenerateConfig(BaseModel):
    provider: Literal["openai", "anthropic", "ollama", "fake"] = "openai"
    model: str = "gpt-5.6-sol"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    max_tokens: int = Field(default=4096, ge=256)
    ollama_model: str = "gemma2:2b"
    ollama_host: str = "http://localhost:11434"
    prompt_version: str = "v1"
    # One regeneration when citation validation fails, then refuse.
    max_regenerations: int = Field(default=1, ge=0, le=3)


class GuardrailConfig(BaseModel):
    max_query_chars: int = Field(default=500, ge=1)
    min_query_chars: int = Field(default=3, ge=1)
    # Cosine similarity to the corpus centroid below which a query is out of scope.
    scope_threshold: float = Field(default=0.18, ge=-1.0, le=1.0)
    # Best retrieved score below which we refuse instead of calling the model.
    relevance_floor: float = Field(default=0.25, ge=-1.0, le=1.0)
    # Chunks whose containment coefficient (shared shingles / smaller chunk) against
    # a higher-ranked chunk reaches this are dropped as near-duplicates. 0.8 fires on
    # split-part overlap and verbatim quoting without collapsing merely-similar text.
    dedup_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    scan_retrieved_for_injection: bool = True
    require_citations: bool = True
    # Quotes must appear verbatim in their cited chunk. Turning this off is a footgun.
    verify_quotes: bool = True
    # Minimum quote length that counts as evidence.
    min_quote_chars: int = Field(default=12, ge=1)
    redact_pii_in_output: bool = True


class EvalConfig(BaseModel):
    judge_provider: Literal["openai", "anthropic", "ollama", "fake"] = "openai"
    judge_model: str = "gpt-5.6-sol"
    judge_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    # Deterministic thresholds that gate CI. Raise these as the system improves.
    min_recall_at_k: float = 0.75
    min_mrr: float = 0.60
    min_citation_validity: float = 0.95
    min_header_boundary_f1: float = 0.85
    max_false_refusal_rate: float = 0.10
    # LLM-judge thresholds, checked in the nightly suite.
    min_faithfulness: float = 0.90
    min_answer_correctness: float = 0.70


class Secrets(BaseSettings):
    """Secrets live only here, only from the environment."""

    model_config = SettingsConfigDict(env_file=str(RepoRoot / ".env"), extra="ignore")

    anthropic_api_key: SecretStr | None = None
    cohere_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None


class Config(BaseModel):
    """The whole system's configuration."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    headers: HeaderConfig = Field(default_factory=HeaderConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> Config:
        """Load from YAML, or return defaults when no file is given."""
        if path is None:
            return cls()
        # Read-then-parse, each wrapped: a typo'd --config path or broken YAML
        # must surface as the package's typed ConfigError naming the path, not a
        # bare FileNotFoundError traceback. Catching OSError on the read (rather
        # than pre-checking existence) also covers permission and is-a-directory
        # errors without a TOCTOU race.
        try:
            text = Path(path).read_text()
        except OSError as exc:
            raise ConfigError(f"cannot read config {path}: {exc}") from exc
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"config {path} is not valid YAML: {exc}") from exc
        return cls.model_validate(data)

    def hash(self) -> str:
        """Stable hash of the config, for run manifests only.

        Paths are excluded: moving the data directory does not invalidate results.
        The OCR disk-cache key is NOT derived from this hash; it is computed
        separately from OcrConfig in ingest/ocr/cached.py, so retrieval or
        generation changes never re-OCR the corpus.
        """
        payload = self.model_dump(mode="json", exclude={"paths"})
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def ingest_hash(self) -> str:
        """Hash of only the stages that affect stored chunks.

        Changing retrieval or generation settings must not force a re-ingest.
        """
        payload = {
            # batch_size is throughput-only and cache merely gates disk-cache
            # lookup; neither changes OCR output, so neither may change this
            # hash. Keep this exclusion set in lockstep with the OCR cache key
            # (_cache_key in ingest/ocr/cached.py), which excludes the same keys
            # for the same reason.
            "ocr": self.ocr.model_dump(mode="json", exclude={"cache", "batch_size"}),
            "headers": self.headers.model_dump(mode="json"),
            "chunk": self.chunk.model_dump(mode="json"),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]
