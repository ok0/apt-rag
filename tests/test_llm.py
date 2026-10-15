import dataclasses

import pytest

from apt_rag.llm import (
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


def test_get_llm_unknown(settings):
    with pytest.raises(LLMError, match="지원하지 않는 APT_RAG_LLM"):
        get_llm(dataclasses.replace(settings, llm_provider="claude"))


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
