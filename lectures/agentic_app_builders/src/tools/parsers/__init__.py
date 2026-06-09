"""Framework-agnostic parsing tools reusable by every agent solution."""

from .html_parser import html_to_text
from .pdf_parser import pdf_to_text, txt_to_text

__all__ = ["html_to_text", "pdf_to_text", "txt_to_text"]
