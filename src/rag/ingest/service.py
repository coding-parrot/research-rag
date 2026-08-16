"""The ingest service: manifest -> PDFs -> OCR -> normalise -> headings -> chunks.

Produces two artifacts:
  - the chunk set (handed to the indexing service)
  - an ingest report per document (headers found, sections split/merged, health)

The report is not decoration. With a single chunking strategy, a paper whose
structure was not detected chunks into one giant blob and silently poisons
retrieval. `IngestReport.unhealthy` is the list to check after every ingest.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from rag.chunking.section import ChunkReport, SectionChunker
from rag.config import Config
from rag.domain import Chunk
from rag.errors import OcrError
from rag.ingest.fetch import PdfFetcher, pin_digests
from rag.ingest.headers import DetectionReport, HeaderDetector, read_outline
from rag.ingest.manifest import Manifest, load_manifest, save_manifest
from rag.ingest.normalize import normalize
from rag.ingest.ocr.base import OcrEngine
from rag.observability import get_logger, timed

log = get_logger("ingest")


@dataclass(frozen=True, slots=True)
class DocumentReport:
    doc_id: str
    ok: bool
    pages: int = 0
    blocks: int = 0
    headers: DetectionReport | None = None
    chunks: ChunkReport | None = None
    error: str = ""

    @property
    def healthy(self) -> bool:
        return (
            self.ok
            and self.headers is not None
            and self.headers.looks_healthy
            and self.chunks is not None
            and self.chunks.looks_healthy
        )


@dataclass(slots=True)
class IngestReport:
    started_at: str
    ingest_hash: str
    documents: list[DocumentReport] = field(default_factory=list)

    @property
    def unhealthy(self) -> list[DocumentReport]:
        return [d for d in self.documents if not d.healthy]

    @property
    def failed(self) -> list[DocumentReport]:
        return [d for d in self.documents if not d.ok]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


class IngestService:
    def __init__(self, config: Config, ocr: OcrEngine, fetcher: PdfFetcher | None = None) -> None:
        self._config = config
        self._ocr = ocr
        self._fetcher = fetcher or PdfFetcher(config.paths.pdfs)
        self._detector = HeaderDetector(config.headers)
        self._chunker = SectionChunker(config.chunk)

    def ingest(
        self,
        manifest: Manifest,
        *,
        paper_ids: Sequence[str] | None = None,
        manifest_path: Path | None = None,
    ) -> tuple[list[Chunk], IngestReport]:
        papers = manifest.select(paper_ids)
        report = IngestReport(
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ingest_hash=self._config.ingest_hash(),
        )
        chunks: list[Chunk] = []

        fetched, fetch_failures = self._fetcher.fetch_all(papers)
        # Pin any digests discovered on first fetch back into the manifest file, so
        # the next run verifies against them instead of trusting the network. Pins
        # go back to the path the manifest was loaded from; the config path is only
        # a fallback for callers that did not say where it came from.
        if any(r.newly_pinned for r in fetched):
            pinned = pin_digests(manifest, fetched)
            save_manifest(pinned, manifest_path or self._config.paths.corpus_manifest)
            log.info("pinned new digests into the manifest")

        for result in fetched:
            doc_report, doc_chunks = self._ingest_one(
                result.paper.id, result.paper.title, result.path
            )
            report.documents.append(doc_report)
            chunks.extend(doc_chunks)

        # A paper that failed to fetch is a failed document, not a silent omission:
        # it must appear in the report and trip the unhealthy exit code downstream.
        for paper_id, error in fetch_failures:
            report.documents.append(DocumentReport(doc_id=paper_id, ok=False, error=error))

        for doc in report.unhealthy:
            log.warning(
                "unhealthy document",
                fields={"doc_id": doc.doc_id, "error": doc.error or "structure not detected"},
            )
        log.info(
            "ingest complete",
            fields={
                "documents": len(report.documents),
                "healthy": len(report.documents) - len(report.unhealthy),
                "chunks": len(chunks),
            },
        )
        return chunks, report

    def _ingest_one(
        self, doc_id: str, title: str, pdf_path: Path
    ) -> tuple[DocumentReport, list[Chunk]]:
        """One document end to end. A failure yields a failed report and no chunks,
        so a broken PDF can never contribute half-built chunks to the corpus."""
        try:
            with timed(log, "ingest.document", doc_id=doc_id):
                ocr = self._ocr.read(pdf_path, doc_id)
                normalized = normalize(ocr, title=title)
                outline = read_outline(pdf_path) if self._config.headers.use_outline else ()
                headings, header_report = self._detector.detect(normalized, outline=outline)

                document = replace(normalized.document, headings=tuple(headings))
                doc_chunks, chunk_report = self._chunker.chunk(document)

                report = DocumentReport(
                    doc_id=doc_id,
                    ok=True,
                    pages=ocr.page_count,
                    blocks=len(ocr.blocks),
                    headers=header_report,
                    chunks=chunk_report,
                )
                return report, list(doc_chunks)
        except OcrError as exc:
            log.error("ocr failed", fields={"doc_id": doc_id, "error": str(exc)})
            return DocumentReport(doc_id=doc_id, ok=False, error=str(exc)), []
        except Exception as exc:
            # The isolation contract above holds for any fault, not just OCR: one
            # broken document must never abort the rest of the corpus mid-run.
            log.error("ingest failed", fields={"doc_id": doc_id, "error": str(exc)})
            return DocumentReport(doc_id=doc_id, ok=False, error=str(exc)), []


def run_ingest(
    config: Config, ocr: OcrEngine, *, paper_ids: Sequence[str] | None = None
) -> tuple[list[Chunk], IngestReport]:
    """Convenience entry point used by the CLI."""
    config.paths.ensure()
    manifest = load_manifest(config.paths.corpus_manifest)
    service = IngestService(config, ocr)
    chunks, report = service.ingest(
        manifest, paper_ids=paper_ids, manifest_path=config.paths.corpus_manifest
    )

    report_path = config.paths.runs / f"ingest-{time.strftime('%Y%m%d-%H%M%S')}.json"
    report_path.write_text(report.to_json())
    log.info("ingest report written", fields={"path": str(report_path)})
    return chunks, report
