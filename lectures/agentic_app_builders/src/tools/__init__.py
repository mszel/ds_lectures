"""Framework-agnostic tools (parsers and summarizer) shared by every agent solution."""

from .parsers import html_to_text, pdf_to_text, txt_to_text
from .summarizer import summarize_for_question

__all__ = ["html_to_text", "pdf_to_text", "txt_to_text", "summarize_for_question"]
