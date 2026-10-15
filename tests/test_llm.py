import dataclasses
from types import SimpleNamespace

import pytest

from apt_rag.llm import (
    ClaudeClient,
    LLMError,
    OllamaClient,
    OllamaConnectionError,
    get_llm,
)


def test_get_llm_ollama(settings):
    client = get_llm(settings)
    assert isinstance(client, OllamaClient)
    assert client.model == "qwen2.5:7b"
    assert client.name == "ollama:qwen2.5:7b"


def test_get_llm_claude(settings, monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = get_llm(dataclasses.replace(settings, llm_provider="claude"))
    assert isinstance(client, ClaudeClient)
    assert client.model == "claude-haiku-5-5"


def test_get_llm_claude_without_key(settings, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        get_llm(dataclasses.replace(settings, llm_provider="claude"))


def test_get_llm_unknown(settings):
    with pytest.raises(LLMError):
        get_llm(dataclasses.replace(settings, llm_provider="gpt"))


def test_ollama_complete_body(monkeypatch):
    client = OllamaClient("http://localhost:11434", "m")
    calls = []

    def fake_request(method, path, body=None):
        calls.append((method, path, body))
        return {"message": {"content": "답"}}

    monkeypatch.setattr(client, "_request", fake_request)
    assert client.complete("S", [{"role": "user", "content": "q"}]) == "답"
    method, path, body = calls[0]
    assert path == "/api/chat"
    assert body["messages"][0] == {"role": "system", "content": "S"}
    assert body["options"]["num_ctx"] == 16384
    assert body["options"]["temperature"] == 0.0
    assert "format" not in body


def test_ollama_complete_connection_error():
    client = OllamaClient("http://127.0.0.1:1", "m")
    with pytest.raises(OllamaConnectionError) as info:
        client.complete("s", [{"role": "user", "content": "q"}])
    assert isinstance(info.value, LLMError)
    assert "ollama serve" in str(info.value)


def test_claude_complete_with_fake_client():
    pytest.importorskip("anthropic")
    recorded = {}

    class FakeMessages:
        def create(self, **kw):
            recorded.update(kw)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="답")])

    fake = SimpleNamespace(messages=FakeMessages())
    client = ClaudeClient("claude-haiku-5-5", client=fake)
    assert client.complete("S", [{"role": "user", "content": "q"}]) == "답"
    assert recorded["system"] == "S"
    assert recorded["model"] == "claude-haiku-5-5"


def test_claude_connection_error_mapped():
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")

    class FakeMessages:
        def create(self, **kw):
            raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))

    fake = SimpleNamespace(messages=FakeMessages())
    client = ClaudeClient("claude-haiku-5-5", client=fake)
    with pytest.raises(LLMError, match="연결"):
        client.complete("S", [{"role": "user", "content": "q"}])
