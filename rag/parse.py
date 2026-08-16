"""Step 1 - Parsing: turn a PDF into text (Week 3: "Data Parsing").

A PDF is not text - it is drawing instructions - so the text layer has to be
extracted. We use PyMuPDF, the same library as the course notebook: its text
extraction handles arXiv PDFs well (reading order, ligatures, spacing). A scanned
document would need OCR instead; the seam is the same either way: file in, text out.
"""

import pymupdf


def parse_pdf(path: str) -> list[str]:
    """Return the text of each page. Index 0 is page 1."""
    doc = pymupdf.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages
