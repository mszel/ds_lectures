"""HTML-to-text parsing tool, shared across all framework notebooks."""

from ._text_clean import clean_raw_text

_STRIP_TAGS = ["script", "style", "nav", "header", "footer", "noscript", "aside", "form"]


def html_to_text(html: str) -> str:
    """
    Convert raw HTML into clean, readable plain text.

    Boilerplate elements (scripts, styles, navigation, headers, footers) are
    removed before the visible text is extracted and whitespace-normalised.

    Args:
        html: Raw HTML markup, e.g. as returned by the scraper MCP ``fetch_url`` tool.

    Returns:
        Cleaned visible page text.

    Example:
        >>> html_to_text("<body><nav>menu</nav><p>Hello</p><script>x()</script></body>")
        'Hello'
    """
    if not isinstance(html, str):
        raise TypeError(f"html must be a str, got {type(html).__name__}")

    from bs4 import BeautifulSoup

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    for tag in soup(_STRIP_TAGS):
        tag.decompose()

    text = soup.get_text(separator="\n")
    return clean_raw_text(text)
