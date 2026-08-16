"""Step 1 - Parsing: turn a PDF into text, including its figures (Week 3: "Data Parsing").

A PDF is not text - it is drawing instructions - so the text layer has to be
extracted. We use PyMuPDF, the same library as the course notebook.

Figures need one extra move (straight from the Week 3 notes: "Images -> Captions"):
an embedding model can only index text, so anything said by a figure is invisible
to retrieval unless it is written down. Almost always the authors already did that -
"Figure 3: Training loss versus steps..." sits in the text layer and gets indexed
like any other sentence. Only when a page has figure images but no such caption do
we ask a vision model for one, appended as "[Figure: ...]".

Captions are cached by image content hash in data/captions.json, so re-ingesting
never pays the vision bill twice.
"""

import base64
import hashlib
import json
from pathlib import Path

import re

import pymupdf
from openai import OpenAI

from rag.embed import api_key

CAPTION_MODEL = "gpt-5.6-luna"  # bulk work: the cheap tier is plenty for captions
MIN_PIXELS = 40_000             # ~200x200: below this it is an icon, not a figure
# An author-written caption already on the page: "Figure 3: ..." / "Fig. 2."
HAS_CAPTION = re.compile(r"^(Figure|Fig\.?)\s*\d+\s*[:.]", re.I | re.M)
CACHE = Path(__file__).parent.parent / "data" / "captions.json"

_client = None
_cache: dict[str, str] | None = None


def parse_pdf(path: str) -> list[str]:
    """Return the text of each page, figure captions appended. Index 0 is page 1."""
    doc = pymupdf.open(path)
    pages = []
    for page in doc:
        text = page.get_text()
        # Authors caption their own figures; vision is the fallback, not the rule.
        if not HAS_CAPTION.search(text):
            for image in _page_figures(doc, page):
                if caption := _caption(image):
                    text += f"\n[Figure: {caption}]\n"
        pages.append(text)
    doc.close()
    _save_cache()
    return pages


def _page_figures(doc: pymupdf.Document, page: pymupdf.Page) -> list[bytes]:
    """PNG bytes of every figure-sized image on the page."""
    figures = []
    for info in page.get_images(full=True):
        pix = pymupdf.Pixmap(doc, info[0])
        if pix.width * pix.height >= MIN_PIXELS:
            if pix.colorspace and pix.colorspace.n > 3:  # CMYK cannot save to PNG
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            figures.append(pix.tobytes("png"))
    return figures


def _caption(png: bytes) -> str:
    """One sentence describing the figure, cached by content hash."""
    global _client
    cache = _load_cache()
    key = hashlib.md5(png).hexdigest()
    if key in cache:
        return cache[key]

    if _client is None:
        _client = OpenAI(api_key=api_key())
    data_url = "data:image/png;base64," + base64.b64encode(png).decode()
    completion = _client.chat.completions.create(
        model=CAPTION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this figure from an ML paper in 1-2 sentences: "
                                          "what it shows and the key takeaway. No preamble."},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
            ],
        }],
        max_completion_tokens=400,
    )
    caption = (completion.choices[0].message.content or "").strip()
    if caption:  # an empty response is not worth caching or inserting
        cache[key] = caption
    return caption


def _load_cache() -> dict[str, str]:
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    return _cache


def _save_cache() -> None:
    if _cache is not None:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(_cache, indent=0))
