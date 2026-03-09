import logging
from collections import Counter
from pathlib import Path

import fitz  # pymupdf
from docx import Document

logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

def _detect_repeated_text(texts: list[str], threshold: float = 0.5) -> set[str]:
    """Return text blocks that appear on more than `threshold` fraction of pages."""
    total = len(texts)
    if total == 0:
        return set()
    counts = Counter(texts)
    return {text for text, count in counts.items() if count / total > threshold}


def _strip_boilerplate(pages: list[dict], repeated: set[str]) -> list[dict]:
    """Remove repeated header/footer lines from each page's text."""
    cleaned = []
    for page in pages:
        lines = page["text"].splitlines()
        filtered = [ln for ln in lines if ln.strip() not in repeated]
        page = {**page, "text": "\n".join(filtered).strip()}
        if page["text"]:
            cleaned.append(page)
    return cleaned


# ── pdf ──────────────────────────────────────────────────────────────────────

def parse_pdf(file_path: str | Path) -> list[dict]:
    """
    Extract text per page from a PDF using PyMuPDF.

    Returns:
        list of {text, metadata: {page_number, source}}
    Skips pages with no extractable text (warns for likely scanned pages).
    Strips repeated headers/footers and preserves table text blocks.
    """
    file_path = Path(file_path)
    pages: list[dict] = []
    first_lines: list[str] = []   # track potential headers
    last_lines: list[str] = []    # track potential footers

    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            # extract_text with "blocks" layout preserves table-like structures
            blocks = page.get_text("blocks", sort=True)  # list of (x0,y0,x1,y1,text,…)
            text = "\n".join(b[4].strip() for b in blocks if b[4].strip())

            if not text:
                logger.warning(
                    "Page %d of '%s' has no extractable text — may be scanned/image-based.",
                    page_num,
                    file_path.name,
                )
                continue

            lines = text.splitlines()
            if lines:
                first_lines.append(lines[0].strip())
                last_lines.append(lines[-1].strip())

            pages.append({
                "text": text,
                "metadata": {
                    "page_number": page_num,
                    "source": file_path.name,
                },
            })

    # detect and strip repeated headers / footers
    repeated = _detect_repeated_text(first_lines) | _detect_repeated_text(last_lines)
    if repeated:
        logger.debug("Stripping repeated boilerplate from '%s': %s", file_path.name, repeated)
        pages = _strip_boilerplate(pages, repeated)

    return pages


# ── docx ─────────────────────────────────────────────────────────────────────

_PARAGRAPHS_PER_SECTION = 30  # group size for page-like sections


def parse_docx(file_path: str | Path) -> list[dict]:
    """
    Extract paragraphs from a DOCX file and group them into page-like sections.

    Returns:
        list of {text, metadata: {page_number, source}}
    Strips repeated headers/footers across sections.
    """
    file_path = Path(file_path)
    doc = Document(file_path)

    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    # chunk paragraphs into fixed-size sections
    raw_sections: list[dict] = []
    for section_idx, start in enumerate(range(0, len(paragraphs), _PARAGRAPHS_PER_SECTION), start=1):
        chunk = paragraphs[start: start + _PARAGRAPHS_PER_SECTION]
        raw_sections.append({
            "text": "\n".join(chunk),
            "metadata": {
                "page_number": section_idx,
                "source": file_path.name,
            },
        })

    if not raw_sections:
        logger.warning("No extractable text found in '%s'.", file_path.name)
        return []

    # detect repeated first/last paragraphs acting as headers/footers
    first_paras = [s["text"].splitlines()[0].strip() for s in raw_sections if s["text"].splitlines()]
    last_paras  = [s["text"].splitlines()[-1].strip() for s in raw_sections if s["text"].splitlines()]
    repeated = _detect_repeated_text(first_paras) | _detect_repeated_text(last_paras)

    if repeated:
        logger.debug("Stripping repeated boilerplate from '%s': %s", file_path.name, repeated)
        raw_sections = _strip_boilerplate(raw_sections, repeated)

    return raw_sections


# ── dispatcher ───────────────────────────────────────────────────────────────

_PARSERS = {
    ".pdf":  parse_pdf,
    ".docx": parse_docx,
}


def parse_document(file_path: str | Path) -> list[dict]:
    """
    Route to the correct parser based on file extension.

    Raises:
        ValueError: if the file type is unsupported.
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    parser = _PARSERS.get(ext)
    if parser is None:
        supported = ", ".join(_PARSERS.keys())
        raise ValueError(
            f"Unsupported file type '{ext}' for '{file_path.name}'. "
            f"Supported types: {supported}"
        )

    logger.info("Parsing '%s' as %s", file_path.name, ext.upper())
    return parser(file_path)