"""PDF- and text-file parsing tools, shared across all framework notebooks."""

import os

from ._text_clean import clean_raw_text


def pdf_to_text(path: str, max_chars: int = 20000) -> str:
    """
    Extract clean text from a local PDF, capped to fit a model context window.

    Args:
        path: Filesystem path to a ``.pdf`` file.
        max_chars: Hard cap on returned characters (default 20000).

    Returns:
        Cleaned document text, truncated to ``max_chars``.

    Example:
        >>> pdf_to_text("paper.pdf", max_chars=500)[:8]  # doctest: +SKIP
        'Abstract'
    """
    if not path.lower().endswith(".pdf"):
        raise ValueError(f"Expected a .pdf path, got {path!r}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    from pypdf import PdfReader

    reader = PdfReader(path)
    raw_text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return clean_raw_text(raw_text)[:max_chars]


def txt_to_text(path: str, max_chars: int = 20000, encoding: str = "utf-8") -> str:
    """
    Read and clean a local plain-text file (mirrors ``pdf_to_text`` for ``.txt`` inputs).

    Args:
        path: Filesystem path to a text file.
        max_chars: Hard cap on returned characters (default 20000).
        encoding: File encoding; undecodable bytes are ignored.

    Returns:
        Cleaned file text, truncated to ``max_chars``.

    Example:
        >>> txt_to_text("notes.txt")[:5]  # doctest: +SKIP
        'Hello'
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Text file not found: {path}")

    with open(path, encoding=encoding, errors="ignore") as handle:
        raw_text = handle.read()
    return clean_raw_text(raw_text)[:max_chars]
