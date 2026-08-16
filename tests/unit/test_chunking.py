import itertools
import string

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rag.chunking.section import SectionChunker
from rag.config import ChunkConfig
from rag.domain import Heading, NormalizedDocument, PageSpan
from rag.errors import ConfigError
from tests.conftest import LOPSIDED_MARKUP, PAPER_MARKUP, make_chunks, make_detected


class TestSectionBoundaries:
    def test_one_chunk_per_section(self):
        chunks = make_chunks(PAPER_MARKUP)
        titles = [c.section_title for c in chunks]
        assert "Introduction" in titles
        assert "Method" in titles
        assert "Experiments" in titles
        assert "Conclusion" in titles

    def test_frontmatter_becomes_a_chunk(self):
        chunks = make_chunks(PAPER_MARKUP)
        front = chunks[0]
        assert front.section_title == "Abstract and frontmatter"
        assert "Selective Attention Networks" in front.text

    def test_section_text_lands_in_its_chunk(self):
        chunks = make_chunks(PAPER_MARKUP)
        method = next(c for c in chunks if c.section_title == "Method")
        assert "gating network" in method.text
        experiments = next(c for c in chunks if c.section_title == "Experiments")
        assert "BLEU" in experiments.text
        assert "gating network is trained jointly" not in experiments.text

    def test_table_stays_with_its_section(self):
        chunks = make_chunks(PAPER_MARKUP)
        experiments = next(c for c in chunks if c.section_title == "Experiments")
        assert "Dense | 27.4" in experiments.text

    def test_page_metadata(self):
        chunks = make_chunks(PAPER_MARKUP)
        intro = next(c for c in chunks if c.section_title == "Introduction")
        experiments = next(c for c in chunks if c.section_title == "Experiments")
        assert intro.page_start == 1
        assert experiments.page_start == 2


class TestDeterminism:
    def test_identical_runs_identical_ids(self):
        first = make_chunks(PAPER_MARKUP)
        second = make_chunks(PAPER_MARKUP)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert [c.text for c in first] == [c.text for c in second]


class TestSizePolicies:
    def test_oversized_section_splits_with_header_inherited(self):
        chunks = make_chunks(LOPSIDED_MARKUP)
        experiment_parts = [c for c in chunks if c.section_title == "Experiments"]
        assert len(experiment_parts) > 1
        for part in experiment_parts:
            assert part.part_count == len(experiment_parts)
            assert part.section_title == "Experiments"
        # Later parts carry the header line so they are self-describing.
        assert experiment_parts[1].text.startswith("2 Experiments")

    def test_parts_respect_budget(self):
        config = ChunkConfig(max_chunk_tokens=128, part_overlap_tokens=16, min_chunk_chars=80)
        chunks = make_chunks(LOPSIDED_MARKUP, chunk_config=config)
        # Header prefix and overlap add slack; 2x budget is the generous bound.
        for chunk in chunks:
            assert len(chunk.text) <= config.max_chunk_chars * 2

    def test_tiny_section_merges_forward(self):
        chunks = make_chunks(LOPSIDED_MARKUP)
        titles = [c.section_title for c in chunks]
        assert "Tiny" in " ".join(titles) or any("Too short." in c.text for c in chunks)
        tiny_direct = [
            c for c in chunks if c.section_title == "Tiny" and c.text.strip() == "Too short."
        ]
        assert not tiny_direct  # it must not survive alone

    def test_split_report(self):
        document = make_detected(LOPSIDED_MARKUP)
        chunker = SectionChunker(
            ChunkConfig(max_chunk_tokens=128, part_overlap_tokens=16, min_chunk_chars=80)
        )
        _, report = chunker.chunk(document)
        assert report.sections_split >= 1
        assert report.sections_merged >= 1
        assert report.looks_healthy


class TestDegenerateInputs:
    def _document(self, text: str, headings: tuple[Heading, ...] = ()) -> NormalizedDocument:
        return NormalizedDocument(
            doc_id="d",
            title="T",
            text=text,
            headings=headings,
            page_spans=(PageSpan(page=1, start=0, end=len(text)),),
        )

    def test_empty_document(self):
        chunker = SectionChunker(ChunkConfig())
        chunks, report = chunker.chunk(self._document(""))
        assert chunks == ()
        assert not report.looks_healthy

    def test_no_headings_single_chunk(self):
        chunker = SectionChunker(ChunkConfig())
        text = "A paragraph with no structure at all. " * 20
        chunks, report = chunker.chunk(self._document(text.strip()))
        assert len(chunks) == 1
        assert report.sections_detected == 1
        assert not report.looks_healthy

    def test_heading_at_offset_zero(self):
        text = "1 Introduction\n\nBody text of the introduction section here."
        heading = Heading(title="Introduction", level=1, char_start=0, page=1, number="1")
        chunker = SectionChunker(ChunkConfig(min_chunk_chars=0))
        chunks, _ = chunker.chunk(self._document(text, (heading,)))
        assert len(chunks) == 1  # no empty frontmatter chunk
        assert chunks[0].section_title == "Introduction"


def _flat_document(*lengths: int) -> NormalizedDocument:
    """Contiguous whitespace-free sections: a frontmatter block, then one numbered
    section per remaining length. No whitespace means offsets survive tightening
    unchanged, so span assertions stay exact."""
    text = ""
    headings: list[Heading] = []
    for i, length in enumerate(lengths):
        if i > 0:
            headings.append(
                Heading(
                    title=f"Sec{i}",
                    level=1,
                    char_start=len(text),
                    page=1,
                    number=str(i),
                    confidence=1.0,
                )
            )
        text += chr(ord("A") + i) * length
    return NormalizedDocument(
        doc_id="d",
        title="T",
        text=text,
        headings=tuple(headings),
        page_spans=(PageSpan(page=1, start=0, end=len(text)),),
    )


class TestMergePolicy:
    def test_short_final_section_folds_backward(self):
        # Frontmatter 50, Sec1 70, Sec2 880, Sec3 100 with floor 200: the
        # undersized FINAL section has no follower, so it must fold into its
        # predecessor instead of surviving alone as retrieval noise.
        document = _flat_document(50, 70, 880, 100)
        chunks, report = SectionChunker(ChunkConfig(min_chunk_chars=200)).chunk(document)
        last = chunks[-1]
        assert last.section_title == "Sec2"
        assert last.char_end == 1100  # the 100-char tail was absorbed
        assert "D" * 100 in last.text
        assert report.sections_merged == 2  # Sec1 forward into frontmatter, Sec3 backward

    def test_short_section_absorbs_at_most_one_follower(self):
        # Frontmatter 50, Sec1 70, Sec2 880 with floor 200. Unlimited chaining
        # used to merge all three into one span titled by the frontmatter,
        # erasing Sec2's identity. The cap emits the still-short merge instead.
        document = _flat_document(50, 70, 880)
        chunks, report = SectionChunker(ChunkConfig(min_chunk_chars=200)).chunk(document)
        assert [c.section_title for c in chunks] == ["Abstract and frontmatter", "Sec2"]
        assert chunks[1].section_number == "2"
        # The capped merge is emitted even though it is still under the floor.
        assert chunks[0].char_end - chunks[0].char_start == 120
        assert report.sections_merged == 1

    def test_report_distinguishes_merge_collapse_from_detection_failure(self):
        # Four detected sections, all under the floor. The report must keep the
        # pre-merge detection count so a merge collapse is not misread as a
        # header detection failure.
        document = _flat_document(40, 40, 40, 40)
        _, report = SectionChunker(ChunkConfig(min_chunk_chars=200)).chunk(document)
        assert report.sections_before_merge == 4
        assert report.sections_detected == 2
        assert report.sections_merged == 2


class TestDuplicateHeadingOffsets:
    def _document(self, headings: tuple[Heading, ...]) -> NormalizedDocument:
        text = "A" * 100 + "B" * 300
        return NormalizedDocument(
            doc_id="d",
            title="T",
            text=text,
            headings=headings,
            page_spans=(PageSpan(page=1, start=0, end=len(text)),),
        )

    def test_higher_confidence_heading_wins_and_drop_is_counted(self):
        # Two headings anchored at the same offset: the higher-confidence one
        # must survive regardless of order, and the drop must be reported
        # instead of happening silently.
        document = self._document(
            (
                Heading(title="First", level=1, char_start=0, page=1, number="1", confidence=0.9),
                Heading(
                    title="Strong", level=1, char_start=100, page=1, number="2", confidence=0.9
                ),
                Heading(title="Weak", level=1, char_start=100, page=1, number="3", confidence=0.4),
            )
        )
        chunks, report = SectionChunker(ChunkConfig(min_chunk_chars=0)).chunk(document)
        titles = [c.section_title for c in chunks]
        assert "Strong" in titles
        assert "Weak" not in titles
        assert report.boundaries_dropped == 1

    def test_equal_confidence_prefers_deeper_heading(self):
        document = self._document(
            (
                Heading(title="Top", level=1, char_start=100, page=1, number="2", confidence=0.8),
                Heading(
                    title="Deep", level=2, char_start=100, page=1, number="2.1", confidence=0.8
                ),
            )
        )
        config = ChunkConfig(max_depth=2, min_chunk_chars=0)
        chunks, report = SectionChunker(config).chunk(document)
        titles = [c.section_title for c in chunks]
        assert "Deep" in titles
        assert "Top" not in titles
        assert report.boundaries_dropped == 1


class TestConfigGuards:
    def test_zero_char_budget_rejected_at_construction(self):
        # 64 tokens at 0.01 chars per token rounds to a 0-char budget, which
        # would leave the split loop unable to advance. Construction must fail
        # loudly instead of ingest hanging later.
        config = ChunkConfig(max_chunk_tokens=64, chars_per_token=0.01)
        assert config.max_chunk_chars == 0
        with pytest.raises(ConfigError):
            SectionChunker(config)


class TestOffsetProvenance:
    """Chunk text and offsets must agree, so provenance is checkable by slicing."""

    def test_text_matches_slice_with_optional_header_prefix(self):
        for markup in (PAPER_MARKUP, LOPSIDED_MARKUP):
            document = make_detected(markup)
            chunks = make_chunks(markup)
            assert chunks
            for chunk in chunks:
                body = document.text[chunk.char_start : chunk.char_end]
                assert body == body.strip()  # offsets are tightened, not the text
                if chunk.part_index == 0:
                    assert chunk.text == body
                else:
                    # Split parts re-attach the section header; the body portion
                    # is still the exact slice.
                    assert chunk.text.endswith(body)
                    assert chunk.text[: -len(body)].endswith("\n\n")

    def test_whitespace_only_parts_dropped_before_numbering(self):
        # A split whose middle part is pure whitespace: surviving parts must be
        # numbered contiguously with an accurate total, or citation labels
        # advertise a part that does not exist.
        text = "A" * 60 + "\n\n" + " " * 70 + "\n\n" + "B" * 40
        document = NormalizedDocument(
            doc_id="d",
            title="T",
            text=text,
            headings=(),
            page_spans=(PageSpan(page=1, start=0, end=len(text)),),
        )
        config = ChunkConfig(
            max_chunk_tokens=64, chars_per_token=1.0, part_overlap_tokens=0, min_chunk_chars=0
        )
        chunks, _ = SectionChunker(config).chunk(document)
        assert len(chunks) == 2
        assert [c.part_index for c in chunks] == [0, 1]
        assert all(c.part_count == 2 for c in chunks)
        bodies = [document.text[c.char_start : c.char_end] for c in chunks]
        assert bodies[0] == "A" * 60
        assert bodies[1] == "B" * 40


# --------------------------------------------------------------------------- #
# Property tests: the invariants that make chunking trustworthy.
# --------------------------------------------------------------------------- #

_words = st.lists(
    st.text(alphabet=string.ascii_lowercase, min_size=2, max_size=9), min_size=30, max_size=250
)


def _synthesise(words: list[str], heading_positions: list[int]) -> str:
    """Build markup with headings sprinkled at the given paragraph positions."""
    paragraphs: list[str] = []
    sentence: list[str] = []
    section = 1
    for word in words:
        sentence.append(word)
        if len(sentence) >= 8:
            paragraphs.append(" ".join(sentence).capitalize() + ".")
            sentence = []
            if len(paragraphs) in heading_positions:
                paragraphs.append(f"# {section}. Section {string.ascii_uppercase[section % 26]}")
                section += 1
    if sentence:
        paragraphs.append(" ".join(sentence).capitalize() + ".")
    return "\n\n".join(paragraphs)


@given(words=_words, positions=st.lists(st.integers(min_value=1, max_value=25), max_size=4))
@settings(max_examples=40, deadline=None)
def test_chunk_offsets_are_monotonic_and_non_overlapping(words, positions):
    markup = _synthesise(words, sorted(set(positions)))
    chunks = make_chunks(markup)

    by_part_zero = [c for c in chunks if c.part_index == 0]
    for left, right in itertools.pairwise(by_part_zero):
        assert left.char_start < right.char_start
        # Non-overlap between *sections*; parts of a split section may overlap by design.
        assert left.char_end <= right.char_start or left.section_title == right.section_title


@given(words=_words, positions=st.lists(st.integers(min_value=1, max_value=25), max_size=4))
@settings(max_examples=40, deadline=None)
def test_no_body_text_is_lost(words, positions):
    """Every non-whitespace character of the document lands in a chunk span.

    Chunk offsets are tightened to the non-whitespace extent of each part, so the
    characters falling outside every span must all be separator whitespace.
    """
    markup = _synthesise(words, sorted(set(positions)))
    document = make_detected(markup)
    chunks = make_chunks(markup)
    if not chunks:
        assert not document.text.strip()
        return

    covered = sorted((c.char_start, c.char_end) for c in chunks)
    cursor = 0
    for start, end in covered:
        if start > cursor:
            assert not document.text[cursor:start].strip()  # gaps hold only whitespace
        cursor = max(cursor, end)
    assert not document.text[cursor:].strip()


@given(words=_words)
@settings(max_examples=20, deadline=None)
def test_chunk_ids_unique(words):
    markup = _synthesise(words, [2, 5])
    chunks = make_chunks(markup)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
