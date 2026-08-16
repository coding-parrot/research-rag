"""Offline half of RAG: papers.yaml -> download -> parse -> chunk -> embed -> index.

Runs once (and again whenever the paper list changes). The online half - answering
questions - lives in rag/pipeline.py and never re-does any of this work.
"""

import urllib.request
from pathlib import Path

import yaml

from rag.chunk import chunk_paper
from rag.parse import parse_pdf
from rag.store import build

PDF_DIR = Path("data/pdfs")


def main() -> None:
    papers = yaml.safe_load(Path("papers.yaml").read_text())
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    chunks = []
    for paper in papers:
        pdf = PDF_DIR / f"{paper['id']}.pdf"
        if not pdf.exists():
            print(f"downloading {paper['id']} ...")
            urllib.request.urlretrieve(paper["url"], pdf)
        pages = parse_pdf(str(pdf))
        paper_chunks = chunk_paper(paper["id"], paper["title"], pages)
        chunks.extend(paper_chunks)
        print(f"{paper['id']:22s} {len(pages):3d} pages -> {len(paper_chunks):3d} chunks")

    build(chunks)
    print(f"\nindexed {len(chunks)} chunks from {len(papers)} papers -> data/index/")


if __name__ == "__main__":
    main()
