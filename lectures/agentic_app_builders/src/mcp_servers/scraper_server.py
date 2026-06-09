"""A minimal web-scraper exposed as an MCP server (stdio transport).

This is the one component deliberately exposed over the Model Context Protocol:
each framework agent spawns it as a subprocess and consumes its ``fetch_url`` /
``fetch_urls`` tools, while the HTML/PDF parsers and summarizer remain ordinary
in-process tools. Run standalone with::

    python -m src.mcp_servers.scraper_server
"""

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("scraper")

_USER_AGENT = "agentic-app-builders/0.1"


def _get(url: str, timeout: float) -> str:
    """Fetch a single URL and return the response body text (raises on non-2xx)."""
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"url must be http(s), got {url!r}")
    response = httpx.get(
        url,
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    )
    response.raise_for_status()
    return response.text


@mcp.tool()
def fetch_url(url: str, timeout: float = 20.0) -> str:
    """
    Fetch a web page and return its raw HTML/text without any parsing.

    Args:
        url: Absolute http(s) URL to GET.
        timeout: Per-request timeout in seconds.

    Returns:
        The response body as text.

    Example:
        fetch_url("https://example.com") -> "<!doctype html>..."
    """
    return _get(url, timeout)


@mcp.tool()
def fetch_urls(urls: list[str], timeout: float = 20.0) -> dict[str, str]:
    """
    Fetch several pages, returning a mapping of URL to its raw HTML/text.

    A failing URL does not abort the batch; its value becomes ``"ERROR: <reason>"``.

    Args:
        urls: Absolute http(s) URLs to GET.
        timeout: Per-request timeout in seconds.

    Returns:
        A dict mapping each input URL to its body text or an error string.

    Example:
        fetch_urls(["https://example.com"]) -> {"https://example.com": "<!doctype html>..."}
    """
    results: dict[str, str] = {}
    for url in urls:
        try:
            results[url] = _get(url, timeout)
        except Exception as error:  # noqa: BLE001 -- reported per-URL, batch continues
            results[url] = f"ERROR: {error}"
    return results


if __name__ == "__main__":
    mcp.run()
