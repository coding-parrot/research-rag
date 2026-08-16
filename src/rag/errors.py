"""Exception hierarchy.

Guardrail outcomes are *not* exceptions: they are `Decision` values. These are for
genuine faults, where continuing would produce a wrong answer rather than a refusal.
"""

from __future__ import annotations


class RagError(Exception):
    """Base for every error this package raises."""


class ConfigError(RagError):
    """Configuration is internally inconsistent or points at something missing."""


class ManifestError(RagError):
    """corpus.yaml is malformed, or an entry failed its integrity check."""


class FetchError(RagError):
    """A document could not be downloaded."""


class OcrError(RagError):
    """The OCR engine failed on a document."""


class HeaderDetectionError(RagError):
    """No usable section structure could be recovered from a document.

    With section chunking as the only strategy, this is fatal for that document
    rather than something to paper over: a paper that silently becomes one giant
    chunk poisons retrieval without ever raising.
    """


class IndexError_(RagError):
    """Vector or lexical index is missing, corrupt, or built with a different model."""


class LlmError(RagError):
    """The language model call failed or returned something unusable."""
