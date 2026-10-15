from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol

from apt_rag.config import Settings


class LLMError(RuntimeError):
    pass


class OllamaError(LLMError):
    pass


class OllamaConnectionError(OllamaError):
    def __init__(self) -> None:
        super().__init__("ollama에 연결할 수 없습니다. `ollama serve` 실행 여부와 APT_RAG_OLLAMA_URL 확인")


class OllamaModelMissingError(OllamaError):
    def __init__(self, model: str) -> None:
        super().__init__(f"모델 '{model}'이(가) 없습니다. `ollama pull {model}` 필요")


class LLM(Protocol):
    name: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str: ...


DEFAULT_NUM_CTX = 16384


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        num_ctx: int = DEFAULT_NUM_CTX,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.name = f"ollama:{model}"

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def check_model(self) -> None:
        try:
            tags = self._request("GET", "/api/tags")
        except urllib.error.URLError as exc:
            raise OllamaConnectionError() from exc
        names = {m.get("name", "") for m in tags.get("models", [])}
        names |= {m.get("model", "") for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise OllamaModelMissingError(self.model)

    def _post_chat(self, body: dict) -> str:
        try:
            response = self._request("POST", "/api/chat", body)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise OllamaModelMissingError(self.model) from exc
            raise OllamaError(f"ollama 요청 실패 (HTTP {exc.code})") from exc
        except urllib.error.URLError as exc:
            raise OllamaConnectionError() from exc
        except TimeoutError as exc:
            raise OllamaError(
                f"ollama 응답 시간 초과 ({self.timeout:.0f}초). 모델 로딩 중이면 잠시 후 다시 시도하세요"
            ) from exc
        return response.get("message", {}).get("content", "")

    def chat(self, messages: list[dict], seed: int) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.7, "seed": seed},
        }
        return self._post_chat(body)

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        return self._post_chat(body)


def get_llm(settings: Settings) -> LLM:
    provider = settings.llm_provider.strip().lower()
    if provider == "ollama":
        return OllamaClient(settings.ollama_url, settings.llm_model)
    raise LLMError(f"지원하지 않는 APT_RAG_LLM: {settings.llm_provider} (현재 ollama만 지원)")
