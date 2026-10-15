from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from apt_rag.chunk import build_units, chunk_units, write_chunks_json
from apt_rag.config import Settings
from apt_rag.embed import BgeM3Embedder
from apt_rag.extract import extract_pdf, write_extracted_md, write_pages_json
from apt_rag.models import Chunk, ExcludedSegment, ExtractResult, InspectReport, SearchHit, ValidationReport
from apt_rag.ocr import OcrUnavailableError, get_ocr_engine
from apt_rag.pdfinspect import inspect_pdf_file
from apt_rag.search import Searcher
from apt_rag.store import write_index
from apt_rag.validate import render_report_md, validate


def inspect_pdf(pdf: Path, settings: Settings) -> InspectReport:
    report = inspect_pdf_file(pdf)
    out_dir = settings.work_dir / report.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inspect.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def extract(pdf: Path, settings: Settings, report: InspectReport | None = None) -> ExtractResult:
    pdf = Path(pdf)
    if report is None:
        report = inspect_pdf(pdf, settings)
    ocr = get_ocr_engine(settings.ocr)
    if report.ocr_pages and settings.ocr == "none":
        raise OcrUnavailableError(f"OCR 필요 쪽: {report.ocr_pages}. APT_RAG_OCR 설정 필요")
    result = extract_pdf(pdf, report, ocr)
    out_dir = settings.work_dir / report.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_extracted_md(result, out_dir / "extracted.md")
    write_pages_json(report, result, out_dir / "pages.json")
    return result


@dataclass
class PrepareResult:
    doc_id: str
    work_dir: Path
    page_count: int
    extract: ExtractResult
    chunks: list[Chunk]
    excluded: list[ExcludedSegment]
    report: ValidationReport


@dataclass
class IndexResult:
    doc_id: str
    work_dir: Path
    chunk_count: int
    report: ValidationReport
    indexed: bool
    aborted_reason: str | None
    dim: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(pdf: Path, settings: Settings) -> PrepareResult:
    pdf = Path(pdf)
    report = inspect_pdf(pdf, settings)
    extract_result = extract(pdf, settings, report)
    units, structure = build_units(extract_result.lines)
    chunks, excluded = chunk_units(units, report.doc_id)
    out_dir = settings.work_dir / report.doc_id
    write_chunks_json(
        out_dir / "chunks.json",
        report.doc_id,
        str(pdf),
        _sha256(pdf),
        chunks,
        excluded,
        structure,
    )
    validation = validate(extract_result, report.page_count, units, chunks, excluded, structure)
    (out_dir / "validation_report.json").write_text(
        json.dumps(asdict(validation), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "validation_report.md").write_text(render_report_md(validation), encoding="utf-8")
    return PrepareResult(
        doc_id=report.doc_id,
        work_dir=out_dir,
        page_count=report.page_count,
        extract=extract_result,
        chunks=chunks,
        excluded=excluded,
        report=validation,
    )


def build_index(
    pdf: Path,
    settings: Settings,
    *,
    no_embed: bool = False,
    strict: bool = False,
    embedder: object | None = None,
) -> IndexResult:
    prep = prepare(pdf, settings)
    if strict and prep.report.has_errors:
        return IndexResult(
            doc_id=prep.doc_id,
            work_dir=prep.work_dir,
            chunk_count=len(prep.chunks),
            report=prep.report,
            indexed=False,
            aborted_reason="검증 오류(--strict)",
        )
    if no_embed:
        return IndexResult(
            doc_id=prep.doc_id,
            work_dir=prep.work_dir,
            chunk_count=len(prep.chunks),
            report=prep.report,
            indexed=False,
            aborted_reason=None,
        )
    embedder = embedder or BgeM3Embedder(settings)
    vectors = embedder.embed_documents([c.embed_text for c in prep.chunks])
    manifest = {
        "doc_id": prep.doc_id,
        "source_pdf": str(pdf),
        "sha256": _sha256(Path(pdf)),
        "embed_model": embedder.model_name,
        "dim": len(vectors[0]) if vectors else 0,
        "chunk_count": len(prep.chunks),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_index(settings, prep.chunks, vectors, manifest)
    return IndexResult(
        doc_id=prep.doc_id,
        work_dir=prep.work_dir,
        chunk_count=len(prep.chunks),
        report=prep.report,
        indexed=True,
        aborted_reason=None,
        dim=manifest["dim"],
    )


def search(query: str, settings: Settings, top_n: int = 5, searcher: Searcher | None = None) -> list[SearchHit]:
    searcher = searcher or Searcher(settings)
    return searcher.search(query, top_n=top_n)
