from rag.domain import (
    Action,
    Answer,
    AnswerStatus,
    Chunk,
    Citation,
    Decision,
    Heading,
    Usage,
    first_denial,
    make_chunk_id,
)


class TestChunkId:
    def test_deterministic(self):
        a = make_chunk_id("mamba", 100, 500, 0)
        b = make_chunk_id("mamba", 100, 500, 0)
        assert a == b

    def test_sensitive_to_every_field(self):
        base = make_chunk_id("mamba", 100, 500, 0)
        assert make_chunk_id("bert", 100, 500, 0) != base
        assert make_chunk_id("mamba", 101, 500, 0) != base
        assert make_chunk_id("mamba", 100, 501, 0) != base
        assert make_chunk_id("mamba", 100, 500, 1) != base

    def test_shape(self):
        chunk_id = make_chunk_id("d", 0, 1, 0)
        assert len(chunk_id) == 16
        int(chunk_id, 16)  # valid hex


class TestChunk:
    def _chunk(self, **overrides):
        defaults = dict(
            chunk_id="abc",
            doc_id="mamba",
            doc_title="Mamba",
            text="body",
            char_start=0,
            char_end=4,
            section_title="Selective Scan",
            page_start=7,
            page_end=7,
            section_number="3.2",
        )
        defaults.update(overrides)
        return Chunk(**defaults)

    def test_citation_label_single_page(self):
        assert self._chunk().citation_label == "Mamba, section 3.2 Selective Scan, p.7"

    def test_citation_label_page_range(self):
        chunk = self._chunk(page_end=9)
        assert chunk.citation_label.endswith("pp.7-9")

    def test_split_section_label(self):
        chunk = self._chunk(part_index=1, part_count=3)
        assert "(part 2/3)" in chunk.section_label
        assert chunk.was_split

    def test_unnumbered_section(self):
        chunk = self._chunk(section_number=None, section_title="Abstract")
        assert chunk.section_label == "Abstract"


class TestHeading:
    def test_parent_number(self):
        h = Heading(title="Setup", level=2, char_start=0, page=1, number="3.1")
        assert h.parent_number == "3"
        assert h.label == "3.1 Setup"

    def test_top_level_has_no_parent(self):
        h = Heading(title="Method", level=1, char_start=0, page=1, number="3")
        assert h.parent_number is None

    def test_unnumbered_label(self):
        h = Heading(title="Abstract", level=1, char_start=0, page=1)
        assert h.label == "Abstract"


class TestDecision:
    def test_deny_blocks(self):
        d = Decision.deny("rule", "nope")
        assert not d.allowed
        assert d.action is Action.DENY

    def test_modify_allows(self):
        assert Decision.modify("rule", "tweaked").allowed

    def test_first_denial(self):
        decisions = [Decision.allow("a"), Decision.deny("b", "no"), Decision.deny("c", "also no")]
        found = first_denial(decisions)
        assert found is not None and found.rule_id == "b"
        assert first_denial([Decision.allow("a")]) is None


class TestAnswer:
    def test_sources_deduplicate_preserving_order(self):
        answer = Answer(
            status=AnswerStatus.OK,
            text="x",
            citations=(
                Citation(chunk_id="1", quote="q", label="Paper A, section 1"),
                Citation(chunk_id="2", quote="q", label="Paper B, section 2"),
                Citation(chunk_id="3", quote="q", label="Paper A, section 1"),
            ),
        )
        assert answer.sources == ("Paper A, section 1", "Paper B, section 2")

    def test_usage_addition(self):
        total = Usage(input_tokens=10, output_tokens=5, llm_calls=1) + Usage(
            input_tokens=3, output_tokens=2, cache_read_input_tokens=7, llm_calls=1
        )
        assert total == Usage(
            input_tokens=13, output_tokens=7, cache_read_input_tokens=7, llm_calls=2
        )

    def test_status_semantics(self):
        assert AnswerStatus.OK.is_answer
        assert not AnswerStatus.INSUFFICIENT_EVIDENCE.is_answer
