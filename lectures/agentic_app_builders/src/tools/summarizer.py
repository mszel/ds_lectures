"""Question-focused summarizer tool.

Calls an LLM directly through the OpenAI SDK so it works unchanged for both the
hosted OpenAI model and a local Ollama model (via ``base_url``). Long inputs are
handled with a simple token-budgeted map-reduce.
"""

import time

from ..config import LlmConfig, load_llm_config

_SYSTEM_PROMPT = (
    "You are a precise summarizer. Summarize ONLY the parts of the provided content "
    "that are relevant to the user's question. Stay strictly grounded in the content; "
    "if the answer is not present, say so plainly. Be concise."
)
_ENCODING_NAME = "o200k_base"


def summarize_for_question(
    text: str,
    question: str,
    *,
    cfg: LlmConfig | None = None,
    max_tokens_per_chunk: int = 6000,
    max_retries: int = 3,
) -> str:
    """
    Produce a question-focused summary of ``text`` using a single LLM call.

    For inputs that exceed ``max_tokens_per_chunk`` the text is split into token
    chunks, each summarised against the question (map), and the partial summaries
    are summarised once more into the final answer (reduce).

    Args:
        text: Source document or page text to summarize.
        question: The question the summary must stay grounded in.
        cfg: LLM provider config; defaults to :func:`load_llm_config`.
        max_tokens_per_chunk: Token budget per chunk before map-reduce engages.
        max_retries: Attempts per LLM call before giving up.

    Returns:
        A concise, question-focused summary.

    Example:
        >>> summarize_for_question("Long article ...", "What are the key risks?")  # doctest: +SKIP
        'The key risks are ...'
    """
    if not text.strip():
        return "No content was provided to summarize."

    cfg = cfg or load_llm_config()

    import tiktoken
    from openai import OpenAI

    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    encoding = tiktoken.get_encoding(_ENCODING_NAME)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens_per_chunk:
        return _summarize_chunk(client, cfg.model, text, question, max_retries)

    chunks = [
        encoding.decode(tokens[start : start + max_tokens_per_chunk])
        for start in range(0, len(tokens), max_tokens_per_chunk)
    ]
    partials = [
        _summarize_chunk(client, cfg.model, chunk, question, max_retries) for chunk in chunks
    ]
    combined = "\n\n".join(partials)
    return _summarize_chunk(client, cfg.model, combined, question, max_retries)


def _summarize_chunk(client, model: str, content: str, question: str, max_retries: int) -> str:
    """Summarize a single chunk of content against the question, with retries."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nContent:\n{content}"},
    ]
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            return response.choices[0].message.content or ""
        except Exception as error:  # noqa: BLE001 -- surfaced after retries
            last_error = error
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Summarization failed after {max_retries} attempts: {last_error}")
