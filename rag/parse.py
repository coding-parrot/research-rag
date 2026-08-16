"""Step 1 - Parsing: turn a PDF into text (Week 3: "Data Parsing").

A PDF is not text. It is a container of drawing instructions, and the text layer
has to be extracted from it. arXiv papers are born digital, so their text layer is
intact and a fast extractor is enough. A scanned document would need OCR instead;
the extraction step is the same seam either way: file in, text out.
"""

import pypdfium2 as pdfium


def parse_pdf(path: str) -> list[str]:
    """Return the text of each page. Index 0 is page 1."""
    doc = pdfium.PdfDocument(path)
    pages = []
    for page in doc:
        textpage = page.get_textpage()
        pages.append(textpage.get_text_bounded() or "")
        textpage.close()
    doc.close()
    return pages
