import os
from pathlib import Path

import pytest

from apt_rag.config import Settings

REPO_ROOT = Path(__file__).parents[1]


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path / "data",
        embed_model="fake-hash",
        device="cpu",
        batch_size=8,
        max_seq_length=512,
        ocr="none",
    )


@pytest.fixture
def example_pdf():
    path = Path(os.environ.get("APT_RAG_EXAMPLE_PDF", REPO_ROOT / ".request" / "example.pdf"))
    if not path.exists():
        pytest.skip(f"예시 PDF가 없습니다: {path}")
    return path


class HashEmbedder:
    model_name = "fake-hash"
    dim = 256

    def _vec(self, text: str) -> list[float]:
        import re
        import zlib

        s = re.sub(r"\s", "", text)
        v = [0.0] * self.dim
        for i in range(len(s) - 1):
            v[zlib.crc32(s[i : i + 2].encode()) % self.dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5
        if norm == 0:
            return [1.0] + [0.0] * (self.dim - 1)
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def hash_embedder():
    return HashEmbedder()
