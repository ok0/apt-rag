from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    embed_model: str
    device: str
    batch_size: int
    max_seq_length: int
    ocr: str
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b"
    min_vec_sim: float = 0.55
    synonyms_path: Path | None = None
    sim_band: float = 0.05
    band_top_k: int = 20
    llm_provider: str = "ollama"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def chroma_dir(self) -> Path:
        return self.index_dir / "chroma"

    @property
    def collection_name(self) -> str:
        return "bylaws"


def load_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or Path.cwd() / ".env", override=False)
    return Settings(
        data_dir=Path(os.environ.get("APT_RAG_DATA_DIR", "data")).resolve(),
        embed_model=os.environ.get("APT_RAG_EMBED_MODEL", "BAAI/bge-m3"),
        device=os.environ.get("APT_RAG_DEVICE", "auto"),
        batch_size=int(os.environ.get("APT_RAG_BATCH_SIZE", "8")),
        max_seq_length=int(os.environ.get("APT_RAG_MAX_SEQ_LENGTH", "1024")),
        ocr=os.environ.get("APT_RAG_OCR", "none"),
        ollama_url=os.environ.get("APT_RAG_OLLAMA_URL", "http://localhost:11434"),
        llm_model=os.environ.get("APT_RAG_LLM_MODEL", "qwen2.5:7b"),
        min_vec_sim=float(os.environ.get("APT_RAG_MIN_VEC_SIM", "0.55")),
        synonyms_path=Path(os.environ.get("APT_RAG_SYNONYMS", "synonyms.txt")).resolve(),
        sim_band=float(os.environ.get("APT_RAG_SIM_BAND", "0.05")),
        band_top_k=int(os.environ.get("APT_RAG_BAND_TOP_K", "20")),
        llm_provider=os.environ.get("APT_RAG_LLM", "ollama"),
    )


def doc_id_from_path(pdf: Path) -> str:
    return re.sub(r"[^\w\-]+", "_", unicodedata.normalize("NFC", pdf.stem)).strip("_") or "doc"


def resolve_device(setting: str) -> str:
    if setting != "auto":
        return setting
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
