import pytest

from rag.config import HeaderConfig
from rag.domain import Block, BlockType, HeaderSource, OcrDocument
from rag.ingest.headers import HeaderDetector, OutlineEntry, _split_number
from rag.ingest.normalize import normalize
from tests.conftest import PAPER_MARKUP, make_normalized


@pytest.fixture()
def detector() -> HeaderDetector:
    return HeaderDetector(HeaderConfig())


class TestDetection:
    def test_finds_all_sections(self, detector):
        result = make_normalized(PAPER_MARKUP)
        headings, report = detector.detect(result)
        labels = [h.label for h in headings]
        assert "1 Introduction" in labels
        assert "2 Method" in labels
        assert "3 Experiments" in labels
        assert "4 Conclusion" in labels
        assert report.looks_healthy

    def test_layout_and_regex_agree_and_boost_confidence(self, detector):
        result = make_normalized(PAPER_MARKUP)
        headings, _ = detector.detect(result)
        intro = next(h for h in headings if h.title == "Introduction")
        assert HeaderSource.LAYOUT in intro.sources
        assert HeaderSource.REGEX in intro.sources
        assert intro.confidence > HeaderSource.LAYOUT.trust  # boosted by agreement

    def test_headings_are_anchored_to_offsets(self, detector):
        result = make_normalized(PAPER_MARKUP)
        headings, _ = detector.detect(result)
        text = result.document.text
        for heading in headings:
            assert text[heading.char_start :].lstrip("0123456789. ").startswith(heading.title[:10])

    def test_figure_captions_rejected(self, detector):
        markup = (
            "# 1. Introduction\n\nBody text that is long enough to matter here.\n\n"
            "[caption] Figure 2. An architecture diagram with a numbered caption\n\n"
            "More body text follows the caption in the same section of the paper."
        )
        headings, _ = detector.detect(make_normalized(markup))
        assert all("Figure" not in h.title for h in headings)

    def test_huge_section_numbers_rejected(self, detector):
        markup = (
            "# 1. Introduction\n\nBody.\n\n1995 Was a year in which many things happened at once."
        )
        headings, _ = detector.detect(make_normalized(markup))
        assert all((h.number or "").split(".")[0] != "1995" for h in headings)

    def test_regex_only_heading_below_default_confidence(self):
        # A numbered line inside a TEXT block: regex fires (0.6), layout does not.
        config = HeaderConfig(min_confidence=0.7)
        detector = HeaderDetector(config)
        markup = "Some text.\n\n3. Results appear mid paragraph as plain text here."
        headings, report = detector.detect(make_normalized(markup))
        assert not headings
        assert report.rejected >= 1

    def test_subsections_get_levels(self, detector):
        markup = (
            "# 3. Experiments\n\nTop level body text for the experiments section.\n\n"
            "# 3.1 Setup\n\nSubsection body about the experimental setup details.\n\n"
            "# 3.2 Results\n\nSubsection body describing the headline results table."
        )
        headings, _ = detector.detect(make_normalized(markup))
        by_label = {h.label: h.level for h in headings}
        assert by_label["3 Experiments"] == 1
        assert by_label["3.1 Setup"] == 2

    def test_unhealthy_report_when_no_structure(self, detector):
        markup = "Just one long paragraph with no headings at all, repeated to fill space."
        _, report = detector.detect(make_normalized(markup))
        assert not report.looks_healthy

    def test_wrapped_section_header_yields_one_heading(self, detector):
        # Surya joins recognised lines with a newline, so a wrapped heading block
        # arrives as "3.1 Selective\nScan". The markup language cannot express a
        # multi-line header block, so build the OCR blocks directly.
        blocks = (
            Block(type=BlockType.SECTION_HEADER, text="3 Experiments", page=1, order=0),
            Block(
                type=BlockType.TEXT,
                text="Top level body text for the experiments section of the paper.",
                page=1,
                order=1,
            ),
            Block(type=BlockType.SECTION_HEADER, text="3.1 Selective\nScan", page=1, order=2),
            Block(
                type=BlockType.TEXT,
                text="Subsection body describing the selective scan mechanism here.",
                page=1,
                order=3,
            ),
        )
        ocr = OcrDocument(
            doc_id="wrapped",
            source_path="memory://wrapped.pdf",
            page_count=1,
            blocks=blocks,
            engine="fake",
            engine_version="fake-1",
        )
        headings, _ = detector.detect(normalize(ocr))
        scans = [h for h in headings if "Selective" in h.title]
        assert len(scans) == 1
        (scan,) = scans
        assert scan.title == "Selective Scan"
        assert scan.number == "3.1"
        assert scan.level == 2


class TestOutlineSignal:
    def test_outline_anchors_and_wins(self, detector):
        result = make_normalized(PAPER_MARKUP)
        outline = (
            OutlineEntry(title="1 Introduction", level=1, page=1),
            OutlineEntry(title="2 Method", level=1, page=1),
        )
        headings, report = detector.detect(result, outline=outline)
        intro = next(h for h in headings if h.title == "Introduction")
        assert HeaderSource.OUTLINE in intro.sources
        assert report.used_outline

    def test_unfindable_bookmark_is_dropped(self, detector):
        result = make_normalized(PAPER_MARKUP)
        outline = (OutlineEntry(title="Nonexistent Section Title", level=1, page=1),)
        headings, _ = detector.detect(result, outline=outline)
        assert all(h.title != "Nonexistent Section Title" for h in headings)

    def test_prose_mention_before_heading_does_not_anchor(self, detector):
        # The abstract mentions "the conclusion" long before the real section 5
        # heading. The bookmark must anchor at the full heading line, not at the
        # first substring occurrence, or a trust-1.0 boundary lands mid-abstract.
        markup = (
            "In the conclusion we argue that scaling laws hold across model families.\n\n"
            "# 1. Introduction\n\nIntroduction body text that is long enough to matter.\n\n"
            "@page\n\n"
            "# 5. Conclusion\n\nDynamic pruning is practical and future work extends it."
        )
        result = make_normalized(markup)
        outline = (OutlineEntry(title="Conclusion", level=1, page=2),)
        headings, _ = detector.detect(result, outline=outline)
        conclusions = [h for h in headings if h.title == "Conclusion"]
        assert len(conclusions) == 1
        (conclusion,) = conclusions
        assert HeaderSource.OUTLINE in conclusion.sources
        assert result.document.text[conclusion.char_start :].startswith("5. Conclusion")

    def test_unmatched_bookmark_never_falls_back_document_wide(self, detector):
        # OCR mangled the section 6 heading, so the bookmark cannot be found near
        # its page. It must be dropped, not anchored at the first occurrence of
        # "limitations" inside the abstract by a document-wide search.
        markup = (
            "We discuss limitations of prior work in the abstract at some length.\n\n"
            "# 1. Introduction\n\nIntroduction body text that is long enough to matter.\n\n"
            "@page\n\n"
            "# 6. Limitat1ons\n\nThe recognised heading differs from the bookmark string."
        )
        result = make_normalized(markup)
        outline = (OutlineEntry(title="6 Limitations", level=1, page=2),)
        headings, _ = detector.detect(result, outline=outline)
        assert all(h.title != "Limitations" for h in headings)
        assert all(HeaderSource.OUTLINE not in h.sources for h in headings)

    def test_numbered_bookmark_anchors_at_the_number(self, detector):
        # The search needle is the number-stripped title; the anchor must still
        # land on the "2." prefix, or the stray number token pollutes the tail of
        # the previous chunk and the boundary sits two characters late.
        result = make_normalized(PAPER_MARKUP)
        outline = (OutlineEntry(title="2 Method", level=1, page=1),)
        headings, _ = detector.detect(result, outline=outline)
        method = next(h for h in headings if h.title == "Method")
        assert HeaderSource.OUTLINE in method.sources
        assert result.document.text[method.char_start :].startswith("2. Method")

    def test_anchor_offsets_survive_non_length_preserving_lowercase(self, detector):
        # U+0130 lowercases to two code points, so a search over text.lower()
        # shifts every later offset. char_start must index the real text.
        markup = (
            "İstanbul University authors present this work in the abstract block.\n\n"
            "# 1. Introduction\n\nIntroduction body text that is long enough to matter."
        )
        result = make_normalized(markup)
        outline = (OutlineEntry(title="1 Introduction", level=1, page=1),)
        headings, _ = detector.detect(result, outline=outline)
        intro = next(h for h in headings if h.title == "Introduction")
        assert HeaderSource.OUTLINE in intro.sources
        assert result.document.text[intro.char_start :].startswith("1. Introduction")

    def test_unnumbered_nested_bookmark_keeps_outline_level(self, detector):
        # A nested bookmark whose title carries no number is the one case where
        # the bookmark tree is the only depth signal. Level 1 here would make the
        # subsection a chunk boundary inside its parent at max_depth=1.
        markup = (
            "# 3. Experiments\n\nTop level body text for the experiments section.\n\n"
            "# Selective Scan\n\nSubsection body about the selective scan details."
        )
        result = make_normalized(markup)
        outline = (
            OutlineEntry(title="3 Experiments", level=1, page=1),
            OutlineEntry(title="Selective Scan", level=2, page=1),
        )
        headings, _ = detector.detect(result, outline=outline)
        by_title = {h.title: h for h in headings}
        assert by_title["Experiments"].level == 1  # numbered: depth from the number
        assert by_title["Selective Scan"].level == 2  # unnumbered: depth from the tree


class TestSplitNumber:
    @pytest.mark.parametrize(
        ("raw", "number", "title"),
        [
            ("3.1 Selective Scan", "3.1", "Selective Scan"),
            ("3. Experiments", "3", "Experiments"),
            ("12 Results", "12", "Results"),
            ("Abstract", None, "Abstract"),
            ("2.1.3 Deep Nesting", "2.1.3", "Deep Nesting"),
        ],
    )
    def test_cases(self, raw, number, title):
        assert _split_number(raw) == (number, title)


class TestMergeWindow:
    def test_nearby_detections_cluster(self):
        # The same heading seen by layout (as a block) and regex (in text) at
        # nearly the same offset must produce one heading, not two.
        detector = HeaderDetector(HeaderConfig())
        result = make_normalized(PAPER_MARKUP)
        headings, _ = detector.detect(result)
        labels = [h.label for h in headings]
        assert len(labels) == len(set(labels))
