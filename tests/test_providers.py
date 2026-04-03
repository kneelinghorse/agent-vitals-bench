"""Unit tests for elicitation/providers.py — all API calls mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from elicitation.providers import (
    AnthropicProvider,
    DeepSeekProvider,
    GenerateResult,
    LocalLlamaCppProvider,
    MiniMaxProvider,
    OPENROUTER_MODELS,
    OpenAIProvider,
    OpenRouterProvider,
    get_provider,
    list_providers,
    verify_doi,
)


# ---------------------------------------------------------------------------
# Helpers — build mock OpenAI / Anthropic responses
# ---------------------------------------------------------------------------


def _openai_response(
    content: str = "hello",
    model: str = "test-model",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> SimpleNamespace:
    """Mimics openai ChatCompletion response shape."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def _anthropic_response(
    text: str = "hello",
    model: str = "claude-haiku-4-5-20251001",
    input_tokens: int = 12,
    output_tokens: int = 8,
) -> SimpleNamespace:
    """Mimics anthropic Message response shape."""
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(
        content=[block],
        model=model,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


# ---------------------------------------------------------------------------
# GenerateResult
# ---------------------------------------------------------------------------


class TestGenerateResult:
    def test_fields(self) -> None:
        r = GenerateResult(
            content="hi",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model="m",
            provider="p",
        )
        assert r.content == "hi"
        assert r.total_tokens == 15
        assert r.raw is None

    def test_frozen(self) -> None:
        r = GenerateResult("x", 1, 2, 3, "m", "p")
        with pytest.raises(AttributeError):
            r.content = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LocalLlamaCppProvider
# ---------------------------------------------------------------------------


class TestLocalLlamaCpp:
    def test_forbidden_port_raises(self) -> None:
        with pytest.raises(ValueError, match="8008.*reserved"):
            LocalLlamaCppProvider(base_url="http://localhost:8008/v1")

    def test_allowed_ports(self) -> None:
        for port in (8003, 8004, 8005, 8006):
            p = LocalLlamaCppProvider(base_url=f"http://localhost:{port}/v1")
            assert str(port) in p.name

    @patch("elicitation.providers.openai.OpenAI")
    def test_generate(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(
            content="output", prompt_tokens=20, completion_tokens=10
        )

        p = LocalLlamaCppProvider(base_url="http://localhost:8004/v1", model="qwen3.5-27b")
        result = p.generate("test prompt", system="be helpful")

        assert isinstance(result, GenerateResult)
        assert result.content == "output"
        assert result.prompt_tokens == 20
        assert result.completion_tokens == 10
        assert result.total_tokens == 30

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    @patch("elicitation.providers.openai.OpenAI")
    def test_generate(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response()

        p = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")
        result = p.generate("hello")

        assert result.provider == "openai"
        assert result.content == "hello"
        assert result.total_tokens == 15

    @patch("elicitation.providers.openai.OpenAI")
    def test_no_system(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response()

        p = OpenAIProvider(model="gpt-4o-mini", api_key="sk-test")
        p.generate("hello")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    @patch("elicitation.providers.anthropic.Anthropic")
    def test_generate(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _anthropic_response(
            text="claude says hi", input_tokens=15, output_tokens=10
        )

        p = AnthropicProvider(api_key="sk-ant-test")
        result = p.generate("hello", system="be concise")

        assert result.provider == "anthropic"
        assert result.content == "claude says hi"
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 10
        assert result.total_tokens == 25

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["system"] == "be concise"

    @patch("elicitation.providers.anthropic.Anthropic")
    def test_no_system(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create.return_value = _anthropic_response()

        p = AnthropicProvider(api_key="sk-ant-test")
        p.generate("hello")

        call_kwargs = mock_client.messages.create.call_args
        assert "system" not in call_kwargs.kwargs


# ---------------------------------------------------------------------------
# OpenRouterProvider
# ---------------------------------------------------------------------------


class TestOpenRouterProvider:
    @patch("elicitation.providers.openai.OpenAI")
    def test_generate(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(
            model="qwen/qwen3.5-72b"
        )

        p = OpenRouterProvider(api_key="sk-or-test")
        result = p.generate("hello")

        assert result.provider == "openrouter"
        assert result.model == "qwen/qwen3.5-72b"

        # Verify base_url was set to openrouter
        init_kwargs = mock_cls.call_args
        assert "openrouter.ai" in init_kwargs.kwargs["base_url"]


# ---------------------------------------------------------------------------
# MiniMaxProvider
# ---------------------------------------------------------------------------


class TestMiniMaxProvider:
    @patch("elicitation.providers.openai.OpenAI")
    def test_generate(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(
            content="minimax output", model="MiniMax-Text-01",
            prompt_tokens=15, completion_tokens=8,
        )

        p = MiniMaxProvider(api_key="mm-test")
        result = p.generate("hello", system="be helpful")

        assert result.provider == "minimax"
        assert result.content == "minimax output"
        assert result.prompt_tokens == 15
        assert result.completion_tokens == 8
        assert result.total_tokens == 23

        # Verify base_url was set to MiniMax
        init_kwargs = mock_cls.call_args
        assert "minimaxi.chat" in init_kwargs.kwargs["base_url"]

    @patch("elicitation.providers.openai.OpenAI")
    def test_no_system(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response()

        p = MiniMaxProvider(api_key="mm-test")
        p.generate("hello")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"


# ---------------------------------------------------------------------------
# DeepSeekProvider
# ---------------------------------------------------------------------------


class TestDeepSeekProvider:
    @patch("elicitation.providers.openai.OpenAI")
    def test_generate(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(
            content="deepseek output", model="deepseek-chat",
            prompt_tokens=12, completion_tokens=6,
        )

        p = DeepSeekProvider(api_key="ds-test")
        result = p.generate("hello")

        assert result.provider == "deepseek"
        assert result.content == "deepseek output"
        assert result.total_tokens == 18

        # Verify base_url was set to DeepSeek
        init_kwargs = mock_cls.call_args
        assert "deepseek.com" in init_kwargs.kwargs["base_url"]

    @patch("elicitation.providers.openai.OpenAI")
    def test_custom_model(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(
            model="deepseek-reasoner"
        )

        p = DeepSeekProvider(model="deepseek-reasoner", api_key="ds-test")
        result = p.generate("hello")
        assert result.model == "deepseek-reasoner"


# ---------------------------------------------------------------------------
# OpenRouterProvider — model presets
# ---------------------------------------------------------------------------


class TestOpenRouterPresets:
    @patch("elicitation.providers.openai.OpenAI")
    def test_preset_anthropic(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(
            model=OPENROUTER_MODELS["anthropic"]
        )

        p = OpenRouterProvider(model="anthropic", api_key="sk-or-test")
        result = p.generate("hello")
        assert "claude" in result.model

    @patch("elicitation.providers.openai.OpenAI")
    def test_preset_gemini(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(
            model=OPENROUTER_MODELS["gemini"]
        )

        p = OpenRouterProvider(model="gemini", api_key="sk-or-test")
        result = p.generate("hello")
        assert "gemini" in result.model

    @patch("elicitation.providers.openai.OpenAI")
    def test_preset_codex(self, mock_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(
            model=OPENROUTER_MODELS["codex"]
        )

        p = OpenRouterProvider(model="codex", api_key="sk-or-test")
        result = p.generate("hello")
        assert "codex" in result.model

    @patch("elicitation.providers.openai.OpenAI")
    def test_full_model_id_passthrough(self, mock_cls: MagicMock) -> None:
        """Non-preset model IDs are passed through unchanged."""
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _openai_response(
            model="custom/model-123"
        )

        p = OpenRouterProvider(model="custom/model-123", api_key="sk-or-test")
        assert p._model == "custom/model-123"


# ---------------------------------------------------------------------------
# Registry — new providers
# ---------------------------------------------------------------------------


class TestNewProviderRegistry:
    def test_list_includes_new_providers(self) -> None:
        names = list_providers()
        assert "minimax" in names
        assert "deepseek" in names
        assert "openrouter-anthropic" in names
        assert "openrouter-gemini" in names
        assert "openrouter-codex" in names

    @patch("elicitation.providers.openai.OpenAI")
    def test_get_minimax(self, mock_cls: MagicMock) -> None:
        p = get_provider("minimax", api_key="mm-test")
        assert isinstance(p, MiniMaxProvider)

    @patch("elicitation.providers.openai.OpenAI")
    def test_get_deepseek(self, mock_cls: MagicMock) -> None:
        p = get_provider("deepseek", api_key="ds-test")
        assert isinstance(p, DeepSeekProvider)

    @patch("elicitation.providers.openai.OpenAI")
    def test_get_openrouter_preset(self, mock_cls: MagicMock) -> None:
        p = get_provider("openrouter-anthropic", api_key="sk-or-test")
        assert isinstance(p, OpenRouterProvider)
        assert "claude" in p._model

    @patch("elicitation.providers.openai.OpenAI")
    def test_get_openrouter_gemini(self, mock_cls: MagicMock) -> None:
        p = get_provider("openrouter-gemini", api_key="sk-or-test")
        assert isinstance(p, OpenRouterProvider)
        assert "gemini" in p._model


# ---------------------------------------------------------------------------
# verify_doi
# ---------------------------------------------------------------------------


class TestVerifyDoi:
    @patch("elicitation.providers.httpx.get")
    def test_valid_doi(self, mock_get: MagicMock) -> None:
        mock_get.return_value = SimpleNamespace(status_code=200)
        assert verify_doi("10.1234/fake", api_key="test-key") is True
        call_kwargs = mock_get.call_args
        assert "DOI:10.1234/fake" in call_kwargs.args[0]
        assert call_kwargs.kwargs["headers"]["x-api-key"] == "test-key"

    @patch("elicitation.providers.httpx.get")
    def test_invalid_doi(self, mock_get: MagicMock) -> None:
        mock_get.return_value = SimpleNamespace(status_code=404)
        assert verify_doi("10.9999/nonexistent") is False

    @patch("elicitation.providers.httpx.get")
    def test_network_error(self, mock_get: MagicMock) -> None:
        import httpx

        mock_get.side_effect = httpx.ConnectError("timeout")
        assert verify_doi("10.1234/fake") is False


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_list_providers(self) -> None:
        names = list_providers()
        assert "openai" in names
        assert "anthropic" in names
        assert "openrouter" in names
        assert "minimax" in names
        assert "deepseek" in names
        assert "qwen3.5-27b" in names
        assert "qwen3.5-9b-a" in names

    def test_get_api_provider(self) -> None:
        with patch("elicitation.providers.openai.OpenAI"):
            p = get_provider("openai", api_key="sk-test")
        assert isinstance(p, OpenAIProvider)

    def test_get_local_provider(self) -> None:
        p = get_provider("qwen3.5-27b")
        assert isinstance(p, LocalLlamaCppProvider)
        assert p.name == "qwen3.5-27b"

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent")
