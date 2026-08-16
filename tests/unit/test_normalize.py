import itertools

from rag.domain import BlockType, PageSpan
from rag.ingest.normalize import _page_spans, clean_text, normalize, page_of_offset
from rag.ingest.ocr.fake import build_document


class TestCleanText:
    def test_dehyphenation(self):
        assert clean_text("informa-\ntion retrieval") == "information retrieval"

    def test_keeps_compound_hyphens(self):
        # Uppercase after the hyphen is not a wrap artifact: the hyphen stays and
        # the words are not glued together.
        cleaned = clean_text("state-\nThe next sentence")
        assert "stateThe" not in cleaned
        assert "state-" in cleaned

    def test_soft_newlines_unwrap(self):
        assert clean_text("this line wraps\nmid sentence") == "this line wraps mid sentence"

    def test_hard_breaks_survive(self):
        assert "\n" in clean_text("First sentence ends.\nRow two | of a table")

    def test_unicode_normalisation(self):
        assert clean_text("ﬁne-tuning") == "fine-tuning"  # ligature fi


class TestNormalize:
    def test_page_furniture_dropped(self):
        doc = build_document(
            "d",
            "[pageheader] Proceedings of ICML 2024\n\nReal body text that stays.\n\n[pagefooter] 3",
        )
        result = normalize(doc)
        assert "Proceedings" not in result.document.text
        assert "Real body text" in result.document.text
        assert result.dropped_blocks == 2

    def test_bare_page_numbers_dropped(self):
        doc = build_document("d", "Body text one.\n\n[footnote] 12\n\nBody text two.")
        result = normalize(doc)
        assert "Body text one." in result.document.text
        assert "\n12\n" not in f"\n{result.document.text}\n"

    def test_repeated_boilerplate_dropped(self):
        pages = []
        for i in range(4):
            pages.append(f"Unique content for page {i} that is long enough to stay around.")
            pages.append("[text] Selective Attention Networks Preprint 2024")
            if i < 3:
                pages.append("@page")
        doc = build_document("d", "\n\n".join(pages))
        result = normalize(doc)
        assert result.document.text.count("Preprint 2024") == 0
        assert "Unique content for page 2" in result.document.text

    def test_section_headers_never_dropped_as_boilerplate(self):
        pages = []
        for i in range(4):
            pages.append("# 1. Introduction")
            pages.append(f"Body for page {i} long enough to be kept in the document.")
            if i < 3:
                pages.append("@page")
        doc = build_document("d", "\n\n".join(pages))
        result = normalize(doc)
        assert result.document.text.count("Introduction") == 4

    def test_offsets_are_exact(self):
        doc = build_document("d", "First block here.\n\nSecond block here.")
        result = normalize(doc)
        for span in result.spans:
            assert result.document.text[span.start : span.end] == clean_text(span.block.text)

    def test_span_text_is_the_normalized_slice(self):
        # A soft-wrapped block: the raw OCR text keeps the newline, the span must
        # not. Header detection parses span.text, and a raw newline there breaks
        # number splitting for wrapped headings.
        doc = build_document("d", "informa-\ntion retrieval body text.\n\nSecond block here.")
        result = normalize(doc)
        for span in result.spans:
            assert span.text == result.document.text[span.start : span.end]
        assert result.spans[0].text == "information retrieval body text."
        assert "\n" not in result.spans[0].text

    def test_page_spans_cover_document(self):
        doc = build_document(
            "d", "Page one text.\n\n@page\n\nPage two text.\n\n@page\n\nPage three."
        )
        result = normalize(doc)
        text = result.document.text
        assert result.document.page_at(0) == 1
        assert result.document.page_at(len(text) - 1) == 3
        # Every offset maps to some page; spans are contiguous and non-overlapping.
        spans = result.document.page_spans
        assert spans[0].start == 0
        for left, right in itertools.pairwise(spans):
            assert left.end == right.start

    def test_interleaved_page_bounds_yield_non_overlapping_spans(self):
        # Reading order interleaves the pages: page 1's last block ends at 500 but
        # page 2's first block starts at 450. Offsets 450-499 belong to page 2
        # blocks, so the page 1 span must be clipped at 450, not extended over it.
        spans = _page_spans({1: [0, 500], 2: [450, 900]}, total=900)
        assert spans == (
            PageSpan(page=1, start=0, end=450),
            PageSpan(page=2, start=450, end=900),
        )
        assert page_of_offset(spans, 449) == 1
        assert page_of_offset(spans, 450) == 2
        assert page_of_offset(spans, 499) == 2

    def test_title_inferred_from_title_block(self):
        doc = build_document("d", "[title] A Great Paper\n\nBody text follows here at length.")
        assert normalize(doc).document.title == "A Great Paper"

    def test_explicit_title_wins(self):
        doc = build_document("d", "[title] OCR Title\n\nBody.")
        assert normalize(doc, title="Manifest Title").document.title == "Manifest Title"

    def test_table_blocks_survive(self):
        doc = build_document("d", "Intro text.\n\n[table] Model | BLEU\nDense | 27.4")
        result = normalize(doc)
        assert "Dense | 27.4" in result.document.text
        assert result.spans_of_type(BlockType.TABLE)
