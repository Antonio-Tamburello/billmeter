"""Check what each provider sends, without network: the SDK clients are replaced by fakes."""

import json
from types import SimpleNamespace

import pytest

from billmeter.documents import Part
from billmeter.providers import (
    ProviderConfigError,
    anthropic_provider,
    gemini_provider,
    get_provider,
    ollama_provider,
    openai_provider,
)
from billmeter.schema import json_schema

PDF = Part(b"%PDF-1.4 fake", "application/pdf")
PNG = Part(b"\x89PNG fake", "image/png")


def test_anthropic_asks_for_the_tool_without_forcing_it(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            block = SimpleNamespace(type="tool_use", input={"pod": "IT001E12345678"})
            return SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=10, output_tokens=5))

    fake_sdk = SimpleNamespace(Anthropic=lambda **_: SimpleNamespace(messages=FakeMessages()))
    monkeypatch.setattr(anthropic_provider, "require_package", lambda *_: fake_sdk)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    response = get_provider("anthropic:claude-test").generate([PDF], "prompt", json_schema())
    assert json.loads(response.text) == {"pod": "IT001E12345678"}
    assert captured["tool_choice"] == {"type": "auto"}          # forcing it is a 400 on current models
    assert captured["tools"][0]["name"] == "record_bill_data"
    assert "record_bill_data" in captured["messages"][0]["content"][-1]["text"]
    assert captured["messages"][0]["content"][0]["type"] == "document"
    assert captured["temperature"] == 0.0


def test_openai_sends_pdf_as_file_and_json_schema(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content='{"pdr": null}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                                   usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    def fake_client(**kwargs):
        captured["client"] = kwargs
        return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    monkeypatch.setattr(openai_provider, "require_package", lambda *_: SimpleNamespace(OpenAI=fake_client))
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    get_provider("openai:gpt-test").generate([PDF], "prompt", json_schema())
    assert captured["messages"][0]["content"][1]["type"] == "file"
    assert captured["response_format"]["type"] == "json_schema"
    assert "max_completion_tokens" in captured
    assert captured["client"]["base_url"] is None


def test_openrouter_uses_its_base_url(monkeypatch):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(openai_provider, "require_package", lambda *_: SimpleNamespace(OpenAI=fake_client))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    provider = get_provider("openrouter:some/model")
    assert provider.pdf_native is False
    with pytest.raises(RuntimeError):
        provider.generate([PNG], "prompt", json_schema())
    assert captured["base_url"] == "https://openrouter.ai/api/v1"


def test_compat_requires_base_url(monkeypatch):
    monkeypatch.setattr(openai_provider, "require_package", lambda *_: SimpleNamespace(OpenAI=lambda **_: None))
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    with pytest.raises(ProviderConfigError):
        get_provider("compat:model").generate([PNG], "prompt", json_schema())


def test_missing_key_is_a_config_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(anthropic_provider, "require_package", lambda *_: SimpleNamespace())
    with pytest.raises(ProviderConfigError):
        get_provider("anthropic:x").generate([PDF], "p", {})


def test_gemini_requests_json(monkeypatch):
    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            usage = SimpleNamespace(prompt_token_count=3, candidates_token_count=2)
            return SimpleNamespace(text='{"pod": null}', usage_metadata=usage)

    types = SimpleNamespace(
        Part=SimpleNamespace(from_bytes=lambda data, mime_type: ("part", mime_type)),
        GenerateContentConfig=lambda **kw: kw,
    )
    fake_sdk = SimpleNamespace(types=types, Client=lambda **_: SimpleNamespace(models=FakeModels()))
    monkeypatch.setattr(gemini_provider, "require_package", lambda *_: fake_sdk)
    monkeypatch.setenv("GEMINI_API_KEY", "test")

    response = get_provider("gemini:gemini-test").generate([PDF], "prompt", json_schema())
    assert response.input_tokens == 3
    assert captured["config"]["response_mime_type"] == "application/json"
    assert captured["contents"][0] == ("part", "application/pdf")


def test_ollama_sends_images_and_schema(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured.update(url=url, payload=json)
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {"message": {"content": "{}"}, "prompt_eval_count": 7})

    monkeypatch.setattr(ollama_provider.httpx, "post", fake_post)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    provider = get_provider("ollama:some-vision-model:7b")
    assert provider.model == "some-vision-model:7b" and provider.pdf_native is False
    provider.generate([PNG], "prompt", json_schema())
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["format"]["type"] == "object"
    assert len(captured["payload"]["messages"][0]["images"]) == 1


def test_missing_model_name_is_explained():
    with pytest.raises(ValueError, match="Missing model name"):
        get_provider("ollama:")
