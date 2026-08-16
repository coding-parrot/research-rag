import itertools
import string

from hypothesis import given, settings
from hypothesis import strategies as st

from rag.chunking.section import SectionChunker
from rag.config import ChunkConfig
from rag.domain import Heading, NormalizedDocument, PageSpan
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
    """Every character of the document lands in at least one chunk span."""
    markup = _synthesise(words, sorted(set(positions)))
    document = make_detected(markup)
    chunks = make_chunks(markup)
    if not chunks:
        assert not document.text.strip()
        return

    covered = sorted((c.char_start, c.char_end) for c in chunks)
    cursor = covered[0][0]
    assert cursor == 0
    for start, end in covered:
        assert start <= cursor  # no gap
        cursor = max(cursor, end)
    assert cursor == len(document.text)


@given(words=_words)
@settings(max_examples=20, deadline=None)
def test_chunk_ids_unique(words):
    markup = _synthesise(words, [2, 5])
    chunks = make_chunks(markup)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
