"""Eval dataset schemas and loading.

Two committed dataset files, both YAML:

  golden    questions with reference answers and required citations. Adversarial
            coverage (injection and exfiltration attempts) lives inside this set
            as `category: adversarial` items; there is no separate adversarial
            file.
  headers   hand-labelled section lists for header-detection scoring

A third, optional file holds human-labelled judge calibration examples
(`load_judge_calibration`); when it is absent the runner notes that judge
agreement was not measured rather than failing.

The golden set is bootstrapped by a model and then human-reviewed. `reviewed: true`
is a per-item field, and the runner refuses to gate CI on unreviewed items, which is
what keeps "bootstrapped" from quietly becoming "made up".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from rag.errors import ManifestError


class QuestionCategory(StrEnum):
    FACTUAL = "factual"  # single paper, single section
    MULTI_HOP = "multi_hop"  # needs more than one paper
    TABLE_LOOKUP = "table_lookup"  # the answer lives in a table or figure caption
    DEFINITION = "definition"  # terminology
    UNANSWERABLE = "unanswerable"  # in scope, not covered by the corpus
    OUT_OF_SCOPE = "out_of_scope"  # not about the corpus at all
    ADVERSARIAL = "adversarial"  # injection or exfiltration attempt

    @property
    def expects_answer(self) -> bool:
        return self in {
            QuestionCategory.FACTUAL,
            QuestionCategory.MULTI_HOP,
            QuestionCategory.TABLE_LOOKUP,
            QuestionCategory.DEFINITION,
        }

    @property
    def expects_refusal(self) -> bool:
        return not self.expects_answer


@dataclass(frozen=True, slots=True)
class MustCite:
    """A citation the answer is required to include to count as correct."""

    paper: str  # doc_id from the corpus manifest
    # Numeric targets ("3", "3.2") match the chunk's section NUMBER, exactly or as
    # a dotted prefix ("3" matches 3 and 3.2, never 13.2 or 30). Textual targets
    # substring-match the section title. See metrics.chunk_matches.
    section: str | None = None


@dataclass(frozen=True, slots=True)
class GoldenItem:
    id: str
    question: str
    category: QuestionCategory
    reference_answer: str = ""
    must_cite: tuple[MustCite, ...] = ()
    reviewed: bool = False
    notes: str = ""

    @property
    def gateable(self) -> bool:
        """Only human-reviewed items may gate CI."""
        return self.reviewed


@dataclass(frozen=True, slots=True)
class HeaderLabelItem:
    """Ground truth for one paper's section structure."""

    doc_id: str
    sections: tuple[str, ...]  # ordered top-level section labels, e.g. "3 Experiments"
    notes: str = ""


@dataclass(frozen=True, slots=True)
class GoldenSet:
    items: tuple[GoldenItem, ...]

    def __len__(self) -> int:
        return len(self.items)

    def by_category(self, category: QuestionCategory) -> tuple[GoldenItem, ...]:
        return tuple(i for i in self.items if i.category is category)

    @property
    def reviewed(self) -> tuple[GoldenItem, ...]:
        return tuple(i for i in self.items if i.reviewed)

    @property
    def answerable(self) -> tuple[GoldenItem, ...]:
        return tuple(i for i in self.items if i.category.expects_answer)

    @property
    def refusable(self) -> tuple[GoldenItem, ...]:
        return tuple(i for i in self.items if i.category.expects_refusal)


def load_golden(path: Path | str) -> GoldenSet:
    raw = _load_yaml(path)
    entries = raw.get("questions")
    if not isinstance(entries, list) or not entries:
        raise ManifestError(f"{path} must define a non-empty 'questions' list")

    items: list[GoldenItem] = []
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"questions[{i}] must be a mapping")
        item_id = str(entry.get("id", ""))
        if not item_id:
            raise ManifestError(f"questions[{i}] is missing an id")
        if item_id in seen:
            raise ManifestError(f"duplicate question id {item_id!r}")
        seen.add(item_id)

        try:
            category = QuestionCategory(str(entry.get("category", "")))
        except ValueError as exc:
            valid = ", ".join(c.value for c in QuestionCategory)
            raise ManifestError(
                f"questions[{i}] has unknown category {entry.get('category')!r}; valid: {valid}"
            ) from exc

        question = str(entry.get("question", "")).strip()
        if not question:
            raise ManifestError(f"questions[{i}] has an empty question")

        reference = str(entry.get("reference_answer", "")).strip()
        if category.expects_answer and not reference:
            raise ManifestError(
                f"questions[{i}] ({item_id}): category {category.value} requires a reference_answer"
            )

        raw_cites = entry.get("must_cite", [])
        if not isinstance(raw_cites, list):
            raise ManifestError(f"questions[{i}] ({item_id}): must_cite must be a list")
        must_cite = tuple(_parse_must_cite(c, i, j) for j, c in enumerate(raw_cites))
        if category.expects_answer and not must_cite:
            raise ManifestError(
                f"questions[{i}] ({item_id}): answerable questions must declare must_cite"
            )

        items.append(
            GoldenItem(
                id=item_id,
                question=question,
                category=category,
                reference_answer=reference,
                must_cite=must_cite,
                reviewed=bool(entry.get("reviewed", False)),
                notes=str(entry.get("notes", "")),
            )
        )
    return GoldenSet(items=tuple(items))


def _parse_must_cite(raw: object, item_index: int, cite_index: int) -> MustCite:
    """Parse one must_cite entry, refusing anything malformed.

    Malformed entries used to be silently dropped by an isinstance filter, which
    turned an authoring typo into a permanently weaker requirement.
    """
    where = f"questions[{item_index}].must_cite[{cite_index}]"
    if not isinstance(raw, dict):
        raise ManifestError(f"{where} must be a mapping with a 'paper' key")
    paper = raw.get("paper")
    if not isinstance(paper, str) or not paper.strip():
        raise ManifestError(f"{where} must name a 'paper' (a string doc_id)")
    section = raw.get("section")
    if section is None:
        return MustCite(paper=paper.strip())
    # YAML parses an unquoted `section: 3.10` as the float 3.1 before Python ever
    # sees it, silently turning subsection ten into 3.1. The original text is
    # unrecoverable here, so floats are rejected outright. Ints have no such
    # truncation and coerce losslessly.
    if isinstance(section, float):
        raise ManifestError(
            f"{where}: section {section!r} was parsed as a YAML float, which "
            'truncates values like 3.10; quote it in the YAML (section: "3.10")'
        )
    if isinstance(section, bool) or not isinstance(section, str | int):
        raise ManifestError(f'{where}: section must be a string, e.g. "3.2"')
    text = str(section).strip()
    return MustCite(paper=paper.strip(), section=text or None)


def load_header_labels(path: Path | str) -> tuple[HeaderLabelItem, ...]:
    raw = _load_yaml(path)
    entries = raw.get("documents")
    if not isinstance(entries, list) or not entries:
        raise ManifestError(f"{path} must define a non-empty 'documents' list")

    items: list[HeaderLabelItem] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("doc_id"):
            raise ManifestError(f"documents[{i}] must be a mapping with a doc_id")
        sections = tuple(str(s).strip() for s in entry.get("sections", []) if str(s).strip())
        if len(sections) < 2:
            raise ManifestError(
                f"documents[{i}] ({entry['doc_id']}): needs at least 2 labelled sections"
            )
        items.append(
            HeaderLabelItem(
                doc_id=str(entry["doc_id"]),
                sections=sections,
                notes=str(entry.get("notes", "")),
            )
        )
    return tuple(items)


_JUDGE_VERDICTS = ("faithful", "unfaithful")


def load_judge_calibration(path: Path | str) -> tuple[dict[str, str], ...]:
    """Load human-labelled examples for Judge.calibrate.

    Schema: an `examples` list of {passages, answer, human_verdict} mappings,
    where human_verdict is "faithful" or "unfaithful". Deliberately tiny: the
    point is a spot-check agreement rate, not a second golden set. The runner
    treats a missing file as "agreement not measured", so this loader only ever
    sees paths that exist; a malformed committed file still fails loudly.
    """
    raw = _load_yaml(path)
    entries = raw.get("examples")
    if not isinstance(entries, list) or not entries:
        raise ManifestError(f"{path} must define a non-empty 'examples' list")
    examples: list[dict[str, str]] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"examples[{i}] must be a mapping")
        passages = str(entry.get("passages", "")).strip()
        answer = str(entry.get("answer", "")).strip()
        verdict = str(entry.get("human_verdict", "")).strip()
        if not passages or not answer:
            raise ManifestError(f"examples[{i}] needs non-empty 'passages' and 'answer'")
        if verdict not in _JUDGE_VERDICTS:
            raise ManifestError(
                f"examples[{i}] has human_verdict {entry.get('human_verdict')!r}; "
                f"valid: {', '.join(_JUDGE_VERDICTS)}"
            )
        examples.append({"passages": passages, "answer": answer, "human_verdict": verdict})
    return tuple(examples)


def _load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"eval dataset not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ManifestError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError(f"{path} must contain a mapping at the top level")
    return raw


def coverage_report(golden: GoldenSet) -> dict[str, int]:
    """How many questions per category. The report prints this so a hollowed-out
    category (say, zero table lookups) is visible rather than silently untested."""
    counts: dict[str, int] = {c.value: 0 for c in QuestionCategory}
    for item in golden.items:
        counts[item.category.value] += 1
    counts["total"] = len(golden)
    counts["reviewed"] = len(golden.reviewed)
    return counts


def cited_papers(items: Sequence[GoldenItem]) -> set[str]:
    return {c.paper for item in items for c in item.must_cite}
