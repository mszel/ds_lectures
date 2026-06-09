"""Whitespace and character cleanup shared by the HTML, PDF, and text parsers.

Recovered and modernised from the original ``file_preproc_utils/common_prep_utils.py``
(git history), with mutable default arguments removed and full type hints added.
"""

import re

_DEFAULT_KEEP_CHARS = [
    "@",
    "#",
    "$",
    "%",
    "&",
    "*",
    "(",
    ")",
    "/",
    '"',
    "'",
    "「",
    "」",
    "|",
    "-",
    ":",
    " ",
    ",",
    ".",
    "!",
    "?",
    "[",
    "]",
    "{",
    "}",
    "<",
    ">",
    "=",
    "+",
    "~",
    "`",
    "^",
    ";",
    "\n",
    "。",
    "、",
    "，",
    "\t",
]


def _string_cleaner(in_str: str, spec_chars: list[str] | None = None) -> str:
    """
    Drop non-unicode and non-alphanumeric characters, keeping a punctuation whitelist.

    Args:
        in_str: The raw string to clean.
        spec_chars: Extra characters to preserve beyond the default whitelist.

    Returns:
        The input with disallowed characters removed.

    Example:
        >>> _string_cleaner("a\\x00b c!")
        'ab c!'
    """
    spec_chars = spec_chars or []
    alnum = set(re.findall(r"[^\W]", in_str, re.UNICODE))
    keep = alnum | set(_DEFAULT_KEEP_CHARS) | set(spec_chars)
    return "".join(char for char in in_str if char in keep)


def clean_raw_text(in_str: str, spec_chars: list[str] | None = None) -> str:
    """
    Collapse repeated newlines/tabs/spaces and strip non-unicode noise.

    Args:
        in_str: Raw extracted text (from HTML, a PDF, or a .txt file).
        spec_chars: Extra characters to preserve beyond the default whitelist.

    Returns:
        Cleaned text with at most one blank line between blocks; an empty string
        if nothing meaningful survives. Blocks shorter than two characters are dropped.

    Example:
        >>> clean_raw_text("Hello\\n\\n\\n\\nworld\\t\\tfoo")
        'Hello\\n\\nworld foo'
    """
    spec_chars = spec_chars or []
    collapsed = re.sub(r"\n{2,}", "\n\n", in_str)
    collapsed = re.sub(r"[ \t]+", " ", collapsed)
    blocks = [_string_cleaner(block, spec_chars) for block in collapsed.split("\n\n")]
    blocks = [block for block in blocks if len(block.strip()) >= 2]
    return "\n\n".join(blocks)
