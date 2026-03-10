from unittest.mock import MagicMock

from ouroboros.llm import LLMClient


def test_parse_model_provider_prefixes():
    client = LLMClient(api_key="test-key")

    assert client._parse_model_provider("groq/llama-3.3-70b-versatile") == ("groq", "llama-3.3-70b-versatile")
    assert client._parse_model_provider("nvidia/meta/llama-3.1-70b-instruct") == ("nvidia", "meta/llama-3.1-70b-instruct")
    assert client._parse_model_provider("openrouter/openai/gpt-4.1-mini") == ("openrouter", "openai/gpt-4.1-mini")
    assert client._parse_model_provider("meta-llama/llama-3.3-70b-instruct:free") == (
        "openrouter",
        "meta-llama/llama-3.3-70b-instruct:free",
    )


def test_chat_uses_provider_specific_client_without_openrouter_cost_fetch(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    client = LLMClient(api_key="or-key")

    mock_resp = MagicMock()
    mock_resp.model_dump.return_value = {
        "id": "gen_123",
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        "choices": [{"message": {"content": "ok"}}],
    }

    mock_chat = MagicMock()
    mock_chat.completions.create.return_value = mock_resp
    mock_client = MagicMock()
    mock_client.chat = mock_chat

    monkeypatch.setattr(client, "_get_client", lambda provider: mock_client)

    fetch_calls = []

    def fake_fetch_cost(_generation_id):
        fetch_calls.append(True)
        return 0.01

    monkeypatch.setattr(client, "_fetch_generation_cost", fake_fetch_cost)

    msg, usage = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="groq/llama-3.3-70b-versatile",
    )

    assert msg["content"] == "ok"
    assert usage["total_tokens"] == 13
    assert not fetch_calls, "Generation cost fetch should only happen for OpenRouter requests"
    mock_chat.completions.create.assert_called_once()
    called_model = mock_chat.completions.create.call_args.kwargs["model"]
    assert called_model == "llama-3.3-70b-versatile"


def test_available_models_includes_fallback_provider_models(monkeypatch):
    monkeypatch.setenv("OUROBOROS_MODEL", "groq/llama-3.3-70b-versatile")
    monkeypatch.setenv("OUROBOROS_MODEL_CODE", "nvidia/meta/llama-3.1-70b-instruct")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "openrouter/google/gemini-2.0-flash-001")

    client = LLMClient(api_key="test-key")
    models = client.available_models()

    assert "groq/llama-3.3-70b-versatile" in models
    assert "nvidia/meta/llama-3.1-70b-instruct" in models
    assert "openrouter/google/gemini-2.0-flash-001" in models
