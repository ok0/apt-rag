from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from apt_rag.config import doc_id_from_path
from apt_rag.models import InspectReport, PageInfo

_OK_CHAR_RE = re.compile(r"[가-힣A-Za-z0-9\s\.,()\[\]「」·\-~:;%/①-⑳\"“”'‘’<>〈〉※○]")


def classify_page(page: pymupdf.Page) -> PageInfo:
    text = page.get_text()
    text_chars = len(re.sub(r"\s", "", text))
    covered = sum(
        pymupdf.Rect(info["bbox"]).intersect(page.rect).get_area()
        for info in page.get_image_info()
    )
    image_coverage = min(1.0, covered / page.rect.get_area())
    table_count = len(page.find_tables().tables)

    garbled = False
    if text_chars >= 20:
        chars = re.sub(r"\s", "", text)
        ok_count = len(_OK_CHAR_RE.findall(chars))
        garbled = ok_count / text_chars < 0.5 or text.count("�") >= 5

    if garbled:
        kind = "image"
    elif text_chars >= 20 and image_coverage >= 0.5:
        kind = "mixed"
    elif text_chars >= 20:
        kind = "text"
    elif image_coverage >= 0.3:
        kind = "image"
    else:
        kind = "empty"

    if garbled:
        note = "텍스트 깨짐(OCR 필요)"
    elif kind == "image":
        note = "OCR 필요"
    else:
        note = ""

    return PageInfo(
        page=page.number + 1,
        kind=kind,
        text_chars=text_chars,
        image_coverage=round(image_coverage, 4),
        table_count=table_count,
        garbled=garbled,
        note=note,
    )


def inspect_pdf_file(pdf: Path) -> InspectReport:
    pdf = Path(pdf)
    with pymupdf.open(pdf) as doc:
        pages = [classify_page(page) for page in doc]
    counts: dict[str, int] = {}
    for info in pages:
        counts[info.kind] = counts.get(info.kind, 0) + 1
    ocr_pages = [info.page for info in pages if info.kind == "image"]
    if ocr_pages:
        recommendation = f"{len(ocr_pages)}개 쪽 OCR 필요: {ocr_pages}"
    else:
        recommendation = "모든 쪽이 텍스트 기반: OCR 불필요"
    return InspectReport(
        pdf=str(pdf),
        doc_id=doc_id_from_path(pdf),
        page_count=len(pages),
        pages=pages,
        counts=counts,
        ocr_pages=ocr_pages,
        recommendation=recommendation,
    )
