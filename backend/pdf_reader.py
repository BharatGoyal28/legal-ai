"""PDF text extraction using pdfplumber."""

import re
from pathlib import Path

import pdfplumber


def extract_text(path: str | Path) -> str:
    """Extract all text from a PDF file, normalising whitespace."""
    text_parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    raw = "\n".join(text_parts)
    # collapse multiple blank lines to one
    cleaned = re.sub(r"\n{3,}", "\n\n", raw)
    return cleaned.strip()
