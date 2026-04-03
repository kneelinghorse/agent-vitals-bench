"""Unified LLM provider interface for trace elicitation.

Wraps local llama.cpp instances and API providers behind a common
generate() interface that returns response content + token counts.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx
import openai
import openai.types.chat as _oai_chat
import anthropic


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GenerateResult:
    """Standardized response from any LLM provider."""

    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    provider: str
    raw: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Base provider
# ---------------------------------------------------------------------------

class Provider(ABC):
    """Base class for LLM providers used in elicitation."""

    name: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> GenerateResult:
        """Send a single-turn prompt and return a structured result."""
        ...


# ---------------------------------------------------------------------------
# Local llama.cpp (OpenAI-compatible)
# ---------------------------------------------------------------------------

_FORBIDDEN_PORTS = frozenset({8008})


class LocalLlamaCppProvider(Provider):
    """llama.cpp instance with OpenAI-compatible /v1/chat/completions."""

    def __init__(
        self,
        base_url: str,
        model: str = "local",
        name: str | None = None,
    ) -> None:
        # Safety: reject reserved ports
        from urllib.parse import urlparse

        parsed = urlparse(base_url)
        if parsed.port and parsed.port in _FORBIDDEN_PORTS:
            raise ValueError(
                f"Port {parsed.port} is reserved (news intel cron). "
                "Use one of: 8003, 8004, 8005, 8006."
            )
        self._client = openai.OpenAI(base_url=base_url, api_key="not-needed")
        self._model = model
        self.name = name or f"local-{parsed.port or 'unknown'}"

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> GenerateResult:
        messages: list[_oai_chat.ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return GenerateResult(
            content=choice.message.content or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=resp.model or self._model,
            provider=self.name,
        )


# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------

class OpenAIProvider(Provider):
    """OpenAI API provider."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self._model = model
        self._client = openai.OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> GenerateResult:
        messages: list[_oai_chat.ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return GenerateResult(
            content=choice.message.content or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=resp.model or self._model,
            provider=self.name,
        )


# ---------------------------------------------------------------------------
# Anthropic API
# ---------------------------------------------------------------------------

class AnthropicProvider(Provider):
    """Anthropic Claude API provider."""

    name = "anthropic"

    def __init__(
        self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None
    ) -> None:
        self._model = model
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> GenerateResult:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        resp = self._client.messages.create(**kwargs)
        content = "".join(
            block.text for block in resp.content if block.type == "text"
        )
        return GenerateResult(
            content=content,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            total_tokens=resp.usage.input_tokens + resp.usage.output_tokens,
            model=resp.model,
            provider=self.name,
        )


# ---------------------------------------------------------------------------
# OpenRouter (OpenAI-compatible proxy)
# ---------------------------------------------------------------------------

# OpenRouter model presets for multi-provider corpus collection.
OPENROUTER_MODELS: dict[str, str] = {
    "anthropic": "anthropic/claude-haiku-4-5-20251001",
    "gemini": "google/gemini-2.0-flash-001",
    "codex": "openai/codex-mini-latest",
    "qwen-72b": "qwen/qwen3.5-72b",
}


class OpenRouterProvider(Provider):
    """OpenRouter multi-model proxy.

    Supports model presets via OPENROUTER_MODELS dict:
      - "anthropic" → claude-haiku-4-5-20251001
      - "gemini" → gemini-2.0-flash-001
      - "codex" → codex-mini-latest
      - "qwen-72b" → qwen3.5-72b (default)

    Use via registry: get_provider("openrouter", model="anthropic")
    or get_provider("openrouter-anthropic") for preset shortcuts.
    """

    name = "openrouter"

    def __init__(
        self,
        model: str = "qwen/qwen3.5-72b",
        api_key: str | None = None,
    ) -> None:
        # Resolve preset names to full model IDs
        resolved = OPENROUTER_MODELS.get(model, model)
        self._model = resolved
        self._client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ["OPENROUTER_API_KEY"],
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> GenerateResult:
        messages: list[_oai_chat.ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return GenerateResult(
            content=choice.message.content or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=resp.model or self._model,
            provider=self.name,
        )


# ---------------------------------------------------------------------------
# MiniMax API (OpenAI-compatible)
# ---------------------------------------------------------------------------

class MiniMaxProvider(Provider):
    """MiniMax API provider (OpenAI-compatible endpoint)."""

    name = "minimax"

    def __init__(
        self,
        model: str = "MiniMax-Text-01",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._client = openai.OpenAI(
            base_url="https://api.minimaxi.chat/v1",
            api_key=api_key or os.environ["MINIMAX_API_KEY"],
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> GenerateResult:
        messages: list[_oai_chat.ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return GenerateResult(
            content=choice.message.content or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=resp.model or self._model,
            provider=self.name,
        )


# ---------------------------------------------------------------------------
# DeepSeek API (OpenAI-compatible)
# ---------------------------------------------------------------------------

class DeepSeekProvider(Provider):
    """DeepSeek API provider (OpenAI-compatible endpoint)."""

    name = "deepseek"

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._client = openai.OpenAI(
            base_url="https://api.deepseek.com/v1",
            api_key=api_key or os.environ["DEEPSEEK_API_KEY"],
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> GenerateResult:
        messages: list[_oai_chat.ChatCompletionMessageParam] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return GenerateResult(
            content=choice.message.content or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            model=resp.model or self._model,
            provider=self.name,
        )


# ---------------------------------------------------------------------------
# DOI verification via Semantic Scholar
# ---------------------------------------------------------------------------

def verify_doi(doi: str, api_key: str | None = None) -> bool:
    """Check whether a DOI resolves to a real paper via Semantic Scholar.

    Returns True if the paper exists, False otherwise.
    """
    key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    headers: dict[str, str] = {}
    if key:
        headers["x-api-key"] = key
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    try:
        resp = httpx.get(url, headers=headers, params={"fields": "title"}, timeout=10)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


# ---------------------------------------------------------------------------
# Provider registry + factory
# ---------------------------------------------------------------------------

# Pre-configured local instances (zero-cost first)
LOCAL_ENDPOINTS: dict[str, dict[str, str]] = {
    "qwen3.5-27b": {"base_url": "http://localhost:8004/v1", "model": "qwen3.5-27b"},
    "qwen3.5-9b-a": {"base_url": "http://localhost:8003/v1", "model": "qwen3.5-9b"},
    "qwen3.5-9b-b": {"base_url": "http://localhost:8005/v1", "model": "qwen3.5-9b"},
    "qwen3.5-9b-c": {"base_url": "http://localhost:8006/v1", "model": "qwen3.5-9b"},
}


def get_provider(name: str, **kwargs: Any) -> Provider:
    """Look up a provider by name.

    Local llama.cpp names: qwen3.5-27b, qwen3.5-9b-a, qwen3.5-9b-b, qwen3.5-9b-c
    API names: openai, anthropic, openrouter

    Extra kwargs are forwarded to the provider constructor (e.g. model=...).
    """
    # Local llama.cpp instances
    if name in LOCAL_ENDPOINTS:
        cfg = LOCAL_ENDPOINTS[name]
        return LocalLlamaCppProvider(
            base_url=kwargs.pop("base_url", cfg["base_url"]),
            model=kwargs.pop("model", cfg["model"]),
            name=name,
        )

    # API providers
    registry: dict[str, type[Provider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "openrouter": OpenRouterProvider,
        "minimax": MiniMaxProvider,
        "deepseek": DeepSeekProvider,
    }

    # OpenRouter preset shortcuts: openrouter-anthropic, openrouter-gemini, etc.
    if name.startswith("openrouter-"):
        preset = name[len("openrouter-"):]
        if preset in OPENROUTER_MODELS:
            return OpenRouterProvider(model=preset, **kwargs)

    cls = registry.get(name)
    if cls is None:
        available = sorted(
            set(LOCAL_ENDPOINTS)
            | set(registry)
            | {f"openrouter-{k}" for k in OPENROUTER_MODELS}
        )
        raise ValueError(f"Unknown provider {name!r}. Available: {available}")
    return cls(**kwargs)


def list_providers() -> list[str]:
    """Return all registered provider names."""
    return sorted(
        set(LOCAL_ENDPOINTS)
        | {"openai", "anthropic", "openrouter", "minimax", "deepseek"}
        | {f"openrouter-{k}" for k in OPENROUTER_MODELS}
    )
