from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict
from pathlib import Path

import pymupdf

from apt_rag.models import ExtractResult, InspectReport, Line, RemovedLine, count_content_chars
from apt_rag.ingest.ocr import OcrEngine

PAGE_NUM_RE = re.compile(r"^(?:[-–—]\s*)?\d{1,4}(?:\s*[-–—])?$|^\d{1,4}\s*/\s*\d{1,4}$")
TERMINAL_RE = re.compile(r"(?:[.。:]|다)$")
TOC_LEADER_RE = re.compile(r"(?:·{3,}|\.{4,}|…{2,}|‥{2,})\s*\d*\s*$")
ARTICLE_HEAD_RE = re.compile(r"^제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*\(([^()]{1,60})\)(?=\s|$)(.*)$")
DELETED_RE = re.compile(r"^제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*(?:\(([^()]{0,60})\))?\s*삭\s*제\s*(?:[<〈\[(].*)?$")
CHAPTER_RE = re.compile(r"^제\s*(\d+)\s*장(?:\s+(.{1,40}))?$")
SECTION_RE = re.compile(r"^제\s*(\d+)\s*절(?:\s+(.{1,40}))?$")
ADDENDA_RE = re.compile(r"^부\s*칙(?:\s*[<〈(（\[].*)?$")
APPENDIX_RE = re.compile(r"^[\[〔【]\s*(별표|별지|별첨)\s*([^\]〕】]{0,30})[\]〕】]\s*(.*)$")
PARTICLE_START_RE = re.compile(r"^[에의와과를을로은는이가]")
CIRCLED_RE = re.compile(r"^([①-⑳㉑-㉟㊱-㊿])")
STRUCT_MISC_RE = re.compile(r"^(?:\d{1,2}\.\s|[가-하]\.\s|\(\d+\)|\d+\)|[가-하]\)|[※*•▪○●◦□■▶◇◆]|-\s|<)")
APPENDIX_MARKERS = frozenset({"별표", "별지", "별지서식", "별첨"})


def circled_to_int(ch: str) -> int:
    code = ord(ch)
    if 0x2460 <= code <= 0x2473:
        return code - 0x2460 + 1
    if 0x3251 <= code <= 0x325F:
        return 21 + (code - 0x3251)
    if 0x32B1 <= code <= 0x32BF:
        return 36 + (code - 0x32B1)
    raise ValueError(ch)


def is_appendix_heading(t: str) -> re.Match[str] | None:
    m = APPENDIX_RE.match(t)
    if m and not PARTICLE_START_RE.match(m.group(3)):
        return m
    return None


def is_structural(t: str) -> bool:
    t = t.strip()
    if (
        ARTICLE_HEAD_RE.match(t)
        or DELETED_RE.match(t)
        or CHAPTER_RE.match(t)
        or SECTION_RE.match(t)
        or ADDENDA_RE.match(t)
        or is_appendix_heading(t)
        or CIRCLED_RE.match(t)
        or STRUCT_MISC_RE.match(t)
    ):
        return True
    return re.sub(r"\s", "", t) in APPENDIX_MARKERS


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    parts = [piece.strip() for piece in str(value).split("\n")]
    parts = [piece for piece in parts if piece]
    if not parts:
        return ""
    if all(len(piece) <= 1 for piece in parts):
        joined = "".join(parts)
    else:
        joined = " ".join(parts)
    return joined.replace("|", "\\|")


def table_to_markdown(rows: list[list[object]]) -> str:
    if not rows:
        return ""
    cells = [[_cell_text(c) for c in row] for row in rows]
    width = max(len(row) for row in cells)
    cells = [row + [""] * (width - len(row)) for row in cells]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + "---|" * width]
    for row in cells[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _page_lines(page: pymupdf.Page) -> list[tuple[str, pymupdf.Rect]]:
    result: list[tuple[str, pymupdf.Rect]] = []
    data = page.get_text("dict")
    for block in data["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            text = "".join(span["text"] for span in line["spans"])
            result.append((text, pymupdf.Rect(line["bbox"])))
    return result


def _normalize(raw: str) -> tuple[str, bool, bool, str]:
    raw = unicodedata.normalize("NFC", raw).replace("\xa0", " ").replace("　", " ")
    lead_space = raw[:1].isspace()
    trail_space = raw[-1:].isspace()
    text = re.sub(r"[ \t]+", " ", raw).strip()
    return text, lead_space, trail_space, raw


def _page_entries(page: pymupdf.Page, pno: int, kind: str, ocr: OcrEngine) -> tuple[list[tuple], int]:
    if kind == "image":
        png = page.get_pixmap(dpi=300).tobytes("png")
        text = ocr.ocr_page(png, pno)
        return [("text", t, None) for t in text.split("\n")], 0

    tabs = page.find_tables().tables
    table_boxes = [tuple(tab.bbox) for tab in tabs]
    kept: list[tuple] = []
    for text, rect in _page_lines(page):
        cx = (rect.x0 + rect.x1) / 2
        cy = (rect.y0 + rect.y1) / 2
        inside = any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in table_boxes)
        if not inside:
            kept.append(("text", text, rect))

    for tab in tabs:
        box = tuple(tab.bbox)
        md = table_to_markdown(tab.extract())
        insert_at = len(kept)
        for idx, entry in enumerate(kept):
            if entry[0] == "text" and entry[2].y0 > box[1]:
                insert_at = idx
                break
        kept.insert(insert_at, ("table", md, None))
    return kept, len(tabs)


def _find_running_lines(per_page: list[list[Line]], threshold: float) -> tuple[set[tuple[int, int]], list[RemovedLine]]:
    removed: list[RemovedLine] = []
    candidates: list[tuple[int, int, str, str]] = []
    for page_index, lines in enumerate(per_page):
        nonblank = [i for i, line in enumerate(lines) if line.kind == "text" and line.text]
        count = len(nonblank)
        for order, i in enumerate(nonblank):
            if order < 3:
                position = "header"
            elif order >= count - 3:
                position = "footer"
            else:
                continue
            candidates.append((page_index, i, position, lines[i].text))

    drop: set[tuple[int, int]] = set()
    for page_index, i, _, text in candidates:
        if PAGE_NUM_RE.match(text):
            drop.add((page_index, i))
            removed.append(RemovedLine(page=per_page[page_index][i].page, text=text, reason="page_number"))

    key_pages: dict[str, set[int]] = {}
    keys: dict[tuple[int, int], str] = {}
    for page_index, i, _, text in candidates:
        if (page_index, i) in drop:
            continue
        key = re.sub(r"\d+", "#", re.sub(r"\s", "", text))
        keys[(page_index, i)] = key
        key_pages.setdefault(key, set()).add(page_index)

    for page_index, i, position, text in candidates:
        if (page_index, i) in drop:
            continue
        key = keys[(page_index, i)]
        if len(key_pages[key]) >= threshold and len(key) >= 2:
            drop.add((page_index, i))
            removed.append(RemovedLine(page=per_page[page_index][i].page, text=text, reason=position))
    return drop, removed


def extract_pdf(pdf: Path, report: InspectReport, ocr: OcrEngine) -> ExtractResult:
    pdf = Path(pdf)
    raw_chars = 0
    table_count = 0
    per_page: list[list[Line]] = []

    with pymupdf.open(pdf) as doc:
        for index, page in enumerate(doc):
            pno = index + 1
            raw_chars += count_content_chars(page.get_text())
            kind = report.pages[index].kind
            entries, n_tables = _page_entries(page, pno, kind, ocr)
            table_count += n_tables
            lines: list[Line] = []
            for entry in entries:
                if entry[0] == "table":
                    lines.append(Line(text=entry[1], page=pno, page_end=pno, kind="table"))
                    continue
                text, lead, trail, _ = _normalize(entry[1])
                lines.append(
                    Line(text=text, page=pno, page_end=pno, kind="text", lead_space=lead, trail_space=trail)
                )
            per_page.append(lines)

    text_pages = sum(1 for info in report.pages if info.kind in ("text", "mixed"))
    threshold = max(3, 0.5 * text_pages)

    drop, removed = _find_running_lines(per_page, threshold)

    flat: list[Line] = []
    for page_index, lines in enumerate(per_page):
        out_page: list[Line] = []
        pending_break = False
        for i, line in enumerate(lines):
            if (page_index, i) in drop:
                continue
            if line.kind == "text" and not line.text:
                if out_page:
                    pending_break = True
                continue
            if pending_break:
                line.para_break = True
                pending_break = False
            out_page.append(line)
        flat.extend(out_page)

    out: list[Line] = []
    for line in flat:
        if line.kind == "table" or not out:
            out.append(line)
            continue
        prev = out[-1]
        if (
            prev.kind == "table"
            or line.para_break
            or is_structural(line.text)
            or (line.page != prev.page_end and TERMINAL_RE.search(prev.text))
        ):
            out.append(line)
            continue
        sep = " " if (prev.trail_space or line.lead_space) else ""
        prev.text += sep + line.text
        prev.page_end = line.page
        prev.trail_space = line.trail_space

    extracted_chars = sum(count_content_chars(line.text) for line in out)
    removed_chars = sum(count_content_chars(r.text) for r in removed)
    return ExtractResult(
        doc_id=report.doc_id,
        lines=out,
        removed=removed,
        raw_chars=raw_chars,
        extracted_chars=extracted_chars,
        removed_chars=removed_chars,
        table_count=table_count,
    )


def write_extracted_md(result: ExtractResult, path: Path) -> None:
    out: list[str] = []
    current_page: int | None = None
    for line in result.lines:
        if line.page != current_page:
            out.append("")
            out.append(f"<!-- page {line.page} -->")
            current_page = line.page
        if line.kind == "table":
            out.append("")
            out.append(line.text)
            out.append("")
        else:
            out.append(line.text)
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


def write_pages_json(report: InspectReport, result: ExtractResult, path: Path) -> None:
    data = {
        "pages": [asdict(p) for p in report.pages],
        "removed": [asdict(r) for r in result.removed],
        "stats": {
            "raw_chars": result.raw_chars,
            "extracted_chars": result.extracted_chars,
            "removed_chars": result.removed_chars,
            "table_count": result.table_count,
        },
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
