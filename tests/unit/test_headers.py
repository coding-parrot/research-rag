import pytest

from rag.config import HeaderConfig
from rag.domain import HeaderSource
from rag.ingest.headers import HeaderDetector, OutlineEntry, _split_number
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
