from __future__ import annotations

import functools
import json
from pathlib import Path

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

from apt_rag.models import Chunk

KEEP_TAGS = {"NNG", "NNP", "NR", "NP", "XR", "SL", "SN", "SH", "VV", "VA"}


@functools.lru_cache(1)
def _kiwi() -> Kiwi:
    return Kiwi()


def tokenize(text: str) -> list[str]:
    return [t.form.lower() for t in _kiwi().tokenize(text) if t.tag in KEEP_TAGS]


class BM25Index:
    def __init__(self, chunk_ids: list[str], tokens: list[list[str]]) -> None:
        self.chunk_ids = chunk_ids
        self.tokens = tokens
        self._bm25 = BM25Okapi(tokens)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        return self.search_tokens(tokenize(query), top_k)

    def search_tokens(self, tokens: list[str], top_k: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            ((cid, float(s)) for cid, s in zip(self.chunk_ids, scores) if s > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps({"chunk_ids": self.chunk_ids, "tokens": self.tokens}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["chunk_ids"], data["tokens"])

    @classmethod
    def build(cls, chunks: list[Chunk]) -> BM25Index:
        return cls([c.chunk_id for c in chunks], [tokenize(c.embed_text) for c in chunks])
