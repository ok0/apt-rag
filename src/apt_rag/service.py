from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from apt_rag.config import Settings
from apt_rag.extract import extract_pdf, write_extracted_md, write_pages_json
from apt_rag.models import ExtractResult, InspectReport
from apt_rag.ocr import OcrUnavailableError, get_ocr_engine
from apt_rag.pdfinspect import inspect_pdf_file


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
