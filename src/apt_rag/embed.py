from __future__ import annotations

from typing import Protocol

from apt_rag.config import Settings, resolve_device


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class BgeM3Embedder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.model_name = settings.embed_model
        self._m = None

    def _model(self):
        if self._m is None:
            from sentence_transformers import SentenceTransformer

            m = SentenceTransformer(self.model_name, device=resolve_device(self._settings.device))
            m.max_seq_length = self._settings.max_seq_length
            self._m = m
        return self._m

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model().encode(
            texts,
            batch_size=self._settings.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model().encode(
            [text],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()[0]
