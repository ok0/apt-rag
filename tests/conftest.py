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
