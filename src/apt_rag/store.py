from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
import chromadb.config
from chromadb.errors import NotFoundError

from apt_rag.bm25 import BM25Index
from apt_rag.config import Settings
from apt_rag.models import Chunk

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection

class IndexMismatchError(RuntimeError):
    pass


BATCH = 256


def _client(settings: Settings) -> ClientAPI:
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )


def to_metadata(c: Chunk) -> dict:
    return {
        "doc_id": c.doc_id,
        "part": c.part,
        "scope": c.scope,
        "chapter": c.chapter or "",
        "section": c.section or "",
        "article_ref": c.article_ref,
        "article_no": c.article_no,
        "article_title": c.article_title or "",
        "paragraph_no": c.paragraph_no if c.paragraph_no is not None else 0,
        "appendix_label": c.appendix_label or "",
        "page_start": c.page_start,
        "page_end": c.page_end,
        "header": c.header or "",
    }


def write_index(settings: Settings, chunks: list[Chunk], vectors: list[list[float]], manifest: dict) -> None:
    client = _client(settings)
    try:
        client.delete_collection(settings.collection_name)
    except NotFoundError:
        pass
    collection = client.create_collection(
        settings.collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
    for start in range(0, len(chunks), BATCH):
        part = chunks[start : start + BATCH]
        collection.add(
            ids=[c.chunk_id for c in part],
            embeddings=vectors[start : start + BATCH],
            documents=[c.text for c in part],
            metadatas=[to_metadata(c) for c in part],
        )
    index_dir = settings.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "chunks.json").write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    BM25Index.build(chunks).save(index_dir / "bm25_tokens.json")
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(settings: Settings) -> tuple[Collection, dict[str, Chunk], BM25Index, dict]:
    index_dir = settings.index_dir
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("색인이 없습니다. 먼저 'apt-rag index <pdf>'를 실행하세요.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        collection = _client(settings).get_collection(settings.collection_name, embedding_function=None)
    except (NotFoundError, ValueError) as exc:
        raise FileNotFoundError("색인이 없습니다. 먼저 'apt-rag index <pdf>'를 실행하세요.") from exc
    if collection.count() != manifest["chunk_count"]:
        raise IndexMismatchError("색인 불일치: index를 다시 실행하세요")
    rows = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
    chunks = {r["chunk_id"]: Chunk(**r) for r in rows}
    bm25 = BM25Index.load(index_dir / "bm25_tokens.json")
    return collection, chunks, bm25, manifest
