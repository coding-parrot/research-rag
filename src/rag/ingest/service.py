"""The ingest service: manifest -> documents -> blocks -> normalise -> headings -> chunks.

Two document kinds share one downstream path. PDFs go fetch -> OCR; HTML blogs go
snapshot -> `extract_html`. Both produce an `OcrDocument`, and everything after
that (normalise, header detection, chunking) is the same code, so a blog chunk
and a paper chunk are indistinguishable to retrieval.

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
from rag.errors import FetchError, OcrError
from rag.ingest.fetch import FetchResult, PdfFetcher, pin_digests
from rag.ingest.headers import DetectionReport, HeaderDetector, read_outline
from rag.ingest.html import HtmlSnapshotFetcher, extract_html
from rag.ingest.manifest import Manifest, Paper, PaperKind, load_manifest, save_manifest
from rag.ingest.normalize import normalize
from rag.ingest.ocr.base import OcrEngine
from rag.observability import get_logger, timed

log = get_logger("ingest")

# An engineering blog with an intro and two h2 sections is a perfectly healthy
# document; a research paper with two sections is a detection failure. The html
# floor is therefore lower, and the pdf floor stays whatever the per-stage
# reports say (currently 3).
_HTML_MIN_SECTIONS = 2


@dataclass(frozen=True, slots=True)
class DocumentReport:
    doc_id: str
    ok: bool
    kind: PaperKind = "pdf"
    pages: int = 0
    blocks: int = 0
    headers: DetectionReport | None = None
    chunks: ChunkReport | None = None
    error: str = ""

    @property
    def healthy(self) -> bool:
        """Kind-aware structural health.

        The per-stage `looks_healthy` properties encode research-paper
        expectations. For html the floor drops to `_HTML_MIN_SECTIONS` without
        touching those properties, so pdf semantics are unchanged.
        """
        if not self.ok or self.headers is None or self.chunks is None:
            return False
        if self.kind == "html":
            return (
                self.headers.accepted >= _HTML_MIN_SECTIONS
                and self.chunks.chunks_emitted >= _HTML_MIN_SECTIONS
            )
        return self.headers.looks_healthy and self.chunks.looks_healthy


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
    def __init__(
        self,
        config: Config,
        ocr: OcrEngine,
        fetcher: PdfFetcher | None = None,
        html_fetcher: HtmlSnapshotFetcher | None = None,
    ) -> None:
        self._config = config
        self._ocr = ocr
        self._fetcher = fetcher or PdfFetcher(config.paths.pdfs)
        # Snapshots live at data/html/<id>.html, next to data/pdfs; the manifest
        # comment in corpus.yaml tells humans the same path.
        self._html_fetcher = html_fetcher or HtmlSnapshotFetcher(config.paths.data / "html")
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

        pdf_papers = [p for p in papers if p.kind == "pdf"]
        html_papers = [p for p in papers if p.kind == "html"]
        fetched, fetch_failures = self._fetcher.fetch_all(pdf_papers)
        html_fetched, html_failures = self._fetch_html(html_papers)
        # Pin any digests discovered on first fetch back into the manifest file, so
        # the next run verifies against them instead of trusting the network (or a
        # snapshot someone later re-saves). Pins go back to the path the manifest
        # was loaded from; the config path is only a fallback for callers that did
        # not say where it came from.
        all_fetched = [*fetched, *html_fetched]
        if any(r.newly_pinned for r in all_fetched):
            pinned = pin_digests(manifest, all_fetched)
            save_manifest(pinned, manifest_path or self._config.paths.corpus_manifest)
            log.info("pinned new digests into the manifest")

        for result in fetched:
            doc_report, doc_chunks = self._ingest_one(
                result.paper.id, result.paper.title, result.path
            )
            report.documents.append(doc_report)
            chunks.extend(doc_chunks)

        for result in html_fetched:
            doc_report, doc_chunks = self._ingest_html(result.paper, result.path)
            report.documents.append(doc_report)
            chunks.extend(doc_chunks)

        # A paper that failed to fetch is a failed document, not a silent omission:
        # it must appear in the report and trip the unhealthy exit code downstream.
        for paper_id, error in fetch_failures:
            report.documents.append(DocumentReport(doc_id=paper_id, ok=False, error=error))
        for paper_id, error in html_failures:
            report.documents.append(
                DocumentReport(doc_id=paper_id, ok=False, kind="html", error=error)
            )

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

    def _fetch_html(
        self, papers: Sequence[Paper]
    ) -> tuple[list[FetchResult], list[tuple[str, str]]]:
        """Fetch html snapshots, collecting failures per paper.

        Deliberately NOT `PdfFetcher.fetch_all` semantics: that raises when every
        fetch fails, but on a fresh checkout every snapshot is legitimately
        missing, and aborting the run would take the pdf half of the corpus down
        with it. Each failure becomes a failed document naming its snapshot path.
        """
        results: list[FetchResult] = []
        failures: list[tuple[str, str]] = []
        for paper in papers:
            try:
                results.append(self._html_fetcher.fetch(paper))
            except FetchError as exc:
                failures.append((paper.id, str(exc)))
                log.error("fetch failed", fields={"paper": paper.id, "error": str(exc)})
        return results, failures

    def _ingest_html(self, paper: Paper, path: Path) -> tuple[DocumentReport, list[Chunk]]:
        """One blog end to end, under the same isolation contract as `_ingest_one`.

        Same normalise -> header detection -> chunker path as PDFs, minus the
        outline signal, which only PDFs have.
        """
        try:
            with timed(log, "ingest.document", doc_id=paper.id):
                # errors="replace": a snapshot with a stray non-utf8 byte should
                # cost one replacement character, not the whole document.
                html = path.read_text(encoding="utf-8", errors="replace")
                extracted = extract_html(html, paper.id, str(path))
                normalized = normalize(extracted, title=paper.title)
                headings, header_report = self._detector.detect(normalized)

                document = replace(normalized.document, headings=tuple(headings))
                doc_chunks, chunk_report = self._chunker.chunk(document)

                report = DocumentReport(
                    doc_id=paper.id,
                    ok=True,
                    kind="html",
                    pages=extracted.page_count,
                    blocks=len(extracted.blocks),
                    headers=header_report,
                    chunks=chunk_report,
                )
                return report, list(doc_chunks)
        except Exception as exc:
            log.error("ingest failed", fields={"doc_id": paper.id, "error": str(exc)})
            return DocumentReport(doc_id=paper.id, ok=False, kind="html", error=str(exc)), []


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
