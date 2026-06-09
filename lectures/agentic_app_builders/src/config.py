"""Shared LLM provider configuration.

A single env var, ``LLM_PROVIDER`` (``openai`` or ``ollama``), flips both the
summarizer tool and every framework agent between the hosted OpenAI model and a
local Ollama model exposed over the OpenAI-compatible endpoint.
"""

import os
from dataclasses import dataclass
from typing import Literal

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


@dataclass(frozen=True)
class LlmConfig:
    """
    Provider-agnostic LLM settings consumed everywhere in the project.

    Attributes:
        provider: Either ``"openai"`` or ``"ollama"``.
        model: Bare model name, e.g. ``"gpt-5-mini"`` or ``"qwen3.5:4b"``.
        base_url: OpenAI-compatible endpoint; ``None`` for hosted OpenAI.
        api_key: A real key for OpenAI; any non-empty string for local Ollama.
    """

    provider: Literal["openai", "ollama"]
    model: str
    base_url: str | None
    api_key: str


def load_llm_config() -> LlmConfig:
    """
    Build an :class:`LlmConfig` from environment variables.

    Reads ``LLM_PROVIDER`` (default ``"openai"``). For OpenAI it also reads
    ``OPENAI_MODEL`` and ``OPENAI_API_KEY``; for Ollama it reads ``OLLAMA_MODEL``
    and ``OLLAMA_BASE_URL``.

    Returns:
        A config wired for the selected provider.

    Raises:
        ValueError: If ``LLM_PROVIDER`` is neither ``"openai"`` nor ``"ollama"``.

    Example:
        >>> os.environ["LLM_PROVIDER"] = "ollama"
        >>> load_llm_config().base_url
        'http://localhost:11434/v1'
    """
    provider = os.environ.get("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        return LlmConfig(
            provider="openai",
            model=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            base_url=None,
            api_key=os.environ["OPENAI_API_KEY"],
        )
    if provider == "ollama":
        return LlmConfig(
            provider="ollama",
            model=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            base_url=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            api_key="ollama",
        )
    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}; use 'openai' or 'ollama'.")


def litellm_model_string(cfg: LlmConfig) -> str:
    """
    Return the litellm-style model id used by Google ADK and CrewAI.

    Both providers are OpenAI-compatible, so the ``openai/`` prefix is used for
    each; the local case additionally relies on ``base_url``/``api_key`` being
    passed alongside.

    Args:
        cfg: The active LLM configuration.

    Returns:
        A string such as ``"openai/gpt-5-mini"`` or ``"openai/qwen3.5:4b"``.

    Example:
        >>> litellm_model_string(LlmConfig("openai", "gpt-5-mini", None, "sk-..."))
        'openai/gpt-5-mini'
    """
    return f"openai/{cfg.model}"
