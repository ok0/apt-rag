from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from apt_rag.extract import (
    ADDENDA_RE,
    ARTICLE_HEAD_RE,
    CHAPTER_RE,
    CIRCLED_RE,
    DELETED_RE,
    SECTION_RE,
    TOC_LEADER_RE,
    circled_to_int,
    is_appendix_heading,
    is_structural,
)
from apt_rag.models import Chunk, ExcludedSegment, Line

MAX_CHARS = 1500
ITEM_RE = re.compile(r"^(\d{1,2})\.(?!\d)")

_TAG_RE = re.compile(r"[<〈][^>〉]*[>〉]")
_TAG_OR_SPACE_RE = re.compile(r"[<〈\[][^>〉\]]*[>〉\]]|\s")


def clean_title(s: str | None) -> str:
    s = _TAG_RE.sub("", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split(" ")
    if len(tokens) >= 2 and all(len(t) == 1 for t in tokens):
        return "".join(tokens)
    return s


def article_label(num: int, sub: int | None) -> str:
    return f"제{num}조" + (f"의{sub}" if sub else "")


def article_no_of(unit: Unit) -> str:
    if unit.article_num is None:
        return ""
    return f"{unit.article_num}" + (f"의{unit.article_sub}" if unit.article_sub else "")


def article_ref(unit: Unit) -> str:
    if unit.article_num is not None:
        label = article_label(unit.article_num, unit.article_sub)
        if unit.part == "본문":
            return label
        if unit.part == "부칙":
            return f"부칙 {label}"
        return f"{unit.scope} {label}"
    if unit.appendix_label:
        return unit.appendix_label
    if unit.scope_label:
        return unit.scope_label
    return unit.scope


@dataclass
class Unit:
    part: str
    scope: str
    scope_label: str
    chapter: str
    section: str
    article_num: int | None = None
    article_sub: int | None = None
    article_title: str = ""
    appendix_label: str = ""
    appendix_title: str = ""
    deleted: bool = False
    excluded_reason: str | None = None
    lines: list[Line] = field(default_factory=list)


def _is_chapter_ok(t: str, next_t: str) -> bool:
    return bool(
        CHAPTER_RE.match(t)
        and not TOC_LEADER_RE.search(t)
        and (ARTICLE_HEAD_RE.match(next_t) or DELETED_RE.match(next_t) or SECTION_RE.match(next_t))
    )


def _is_section_ok(t: str, next_t: str) -> bool:
    return bool(
        SECTION_RE.match(t)
        and not TOC_LEADER_RE.search(t)
        and (ARTICLE_HEAD_RE.match(next_t) or DELETED_RE.match(next_t))
    )


def _heading_name(m: re.Match[str], unit_word: str) -> str:
    return f"제{m.group(1)}{unit_word} {clean_title(m.group(2))}".strip()


def _sub(s: str | None) -> int | None:
    return int(s) if s else None


def _appendix_title(rest: str, next_t: str) -> str:
    title = clean_title(rest)
    if not title or title.startswith("("):
        head = "" if is_structural(next_t) or next_t.startswith("|") else clean_title(next_t)
        if rest.startswith("("):
            return (head + " " + clean_title(rest)).strip()
        return head
    return title


class _Builder:
    def __init__(self) -> None:
        self.units: list[Unit] = []
        self.structure: list[Line] = []
        self.part = "preamble"
        self.mode = "doc"
        self.scope = "preamble"
        self.scope_label = ""
        self.chapter = ""
        self.section = ""
        self.addenda_idx = 0
        self.cur: Unit | None = Unit(
            part="preamble",
            scope="preamble",
            scope_label="",
            chapter="",
            section="",
            excluded_reason="preamble",
        )

    def _make(self, **kw: object) -> Unit:
        return Unit(
            part=self.part,
            scope=self.scope,
            scope_label=self.scope_label,
            chapter=self.chapter,
            section=self.section,
            **kw,
        )

    def close(self) -> None:
        if self.cur is not None and self.cur.lines:
            self.units.append(self.cur)
        self.cur = None

    def new_unit(self, **kw: object) -> None:
        self.close()
        self.cur = self._make(**kw)

    def add(self, line: Line) -> None:
        if self.cur is None:
            self.cur = self._make()
        self.cur.lines.append(line)

    def run(self, lines: list[Line]) -> tuple[list[Unit], list[Line]]:
        for i, line in enumerate(lines):
            t = line.text.strip()
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            next_t = nxt.text.strip() if nxt is not None and nxt.kind != "table" else ""
            if line.kind == "table":
                self.add(line)
                continue
            if self.part == "preamble":
                if _is_chapter_ok(t, next_t):
                    self.chapter = _heading_name(CHAPTER_RE.match(t), "장")
                    self.section = ""
                    self.structure.append(line)
                    continue
                if _is_section_ok(t, next_t):
                    self.section = _heading_name(SECTION_RE.match(t), "절")
                    self.structure.append(line)
                    continue
                m = DELETED_RE.match(t) or ARTICLE_HEAD_RE.match(t)
                if m and int(m.group(1)) == 1 and m.group(2) is None and not TOC_LEADER_RE.search(t):
                    self.part = "본문"
                    self.scope = "본문"
                    self.scope_label = ""
                else:
                    self.add(line)
                    continue
            self._body(line, t, next_t)
        self.close()
        return self.units, self.structure

    def _body(self, line: Line, t: str, next_t: str) -> None:
        am = is_appendix_heading(t)
        if am:
            self._appendix(line, am, next_t)
            return
        if self.mode == "form":
            self.add(line)
            return
        if ADDENDA_RE.match(t):
            self.close()
            if self.part == "별첨":
                self.scope = f"{self.scope} 부칙"
            else:
                self.addenda_idx += 1
                self.part = "부칙"
                self.scope = f"부칙#{self.addenda_idx}"
            self.scope_label = clean_title(t)
            self.chapter = ""
            self.section = ""
            self.structure.append(line)
            self.new_unit()
            return
        if _is_chapter_ok(t, next_t):
            self.close()
            self.chapter = _heading_name(CHAPTER_RE.match(t), "장")
            self.section = ""
            self.structure.append(line)
            self.new_unit()
            return
        if _is_section_ok(t, next_t):
            self.close()
            self.section = _heading_name(SECTION_RE.match(t), "절")
            self.structure.append(line)
            self.new_unit()
            return
        if re.sub(r"\s", "", t) in {"별표", "별지", "별지서식", "별첨"}:
            self.structure.append(line)
            return
        dm = DELETED_RE.match(t)
        if dm:
            self.new_unit(
                article_num=int(dm.group(1)),
                article_sub=_sub(dm.group(2)),
                article_title=clean_title(dm.group(3) or ""),
                deleted=True,
            )
            self.add(line)
            return
        hm = ARTICLE_HEAD_RE.match(t)
        if hm:
            title = clean_title(hm.group(3))
            deleted = title == "삭제" or _TAG_OR_SPACE_RE.sub("", hm.group(4)) == "삭제"
            self.new_unit(
                article_num=int(hm.group(1)),
                article_sub=_sub(hm.group(2)),
                article_title=title,
                deleted=deleted,
            )
            self.add(line)
            return
        self.add(line)
        if self.cur is not None and self.cur.article_num is not None and not self.cur.deleted:
            body = "".join(x.text for x in self.cur.lines[1:])
            if _TAG_OR_SPACE_RE.sub("", body) == "삭제":
                self.cur.deleted = True

    def _appendix(self, line: Line, am: re.Match[str], next_t: str) -> None:
        kind = am.group(1)
        inner = re.sub(r"\s+", " ", am.group(2)).strip()
        title = _appendix_title(am.group(3).strip(), next_t)
        self.close()
        self.chapter = ""
        self.section = ""
        mm = re.match(r"^(\d+)\s*(.+)$", inner)
        if kind == "별첨" and re.fullmatch(r"\d+", inner):
            self.part = "별첨"
            self.mode = "doc"
            self.scope = f"별첨 {inner}"
            self.scope_label = f"별첨 {inner} {title}".strip()
            self.new_unit(appendix_label=f"별첨 {inner}", appendix_title=title)
        elif kind == "별첨" and mm:
            self.part = "별첨"
            self.mode = "form"
            self.scope = "별첨서식"
            self.scope_label = ""
            self.new_unit(appendix_label=f"별첨{mm.group(1)} {mm.group(2)}", appendix_title=title)
        else:
            self.part = kind
            self.mode = "form"
            self.scope = kind
            self.scope_label = ""
            self.new_unit(appendix_label=f"{kind} {inner}", appendix_title=title)
        self.add(line)


def build_units(lines: list[Line]) -> tuple[list[Unit], list[Line]]:
    return _Builder().run(lines)


def make_header(
    unit: Unit,
    paragraph_no: int | None,
    split_index: int | None,
    split_total: int | None,
    item_no: int | None = None,
) -> str:
    lab_title = ""
    if unit.article_num is not None:
        label = article_label(unit.article_num, unit.article_sub)
        lab_title = f"{label}({unit.article_title})" if unit.article_title else label
    if unit.article_num is None:
        parts = [f"{unit.appendix_label} {unit.appendix_title}".strip() or unit.scope_label]
    elif unit.part in ("부칙", "별첨"):
        parts = [unit.scope_label, unit.chapter, unit.section, lab_title]
    else:
        parts = [unit.chapter, unit.section, lab_title]
    parts = [p for p in parts if p]
    if paragraph_no is not None:
        parts.append(f"제{paragraph_no}항")
    if item_no is not None:
        parts.append(f"제{item_no}호")
    s = " > ".join(parts)
    if split_total:
        s += f" ({split_index}/{split_total})"
    return f"[{s}]"


def _pack(lines: list[Line], limit: int = MAX_CHARS) -> list[list[Line]]:
    groups: list[list[Line]] = []
    cur: list[Line] = []
    cur_len = 0
    for line in lines:
        n = len(line.text)
        if cur and cur_len + 1 + n > limit:
            groups.append(cur)
            cur = []
            cur_len = 0
        cur_len = n if not cur else cur_len + 1 + n
        cur.append(line)
    if cur:
        groups.append(cur)
    return groups


def _seg_text(lines: list[Line]) -> str:
    return "\n".join(line.text for line in lines)


def _split_paragraphs(lines: list[Line]) -> list[tuple[int | None, list[Line]]]:
    first_no: int | None = None
    head = ARTICLE_HEAD_RE.match(lines[0].text.strip())
    if head:
        m = CIRCLED_RE.match(head.group(4).strip())
        if m:
            first_no = circled_to_int(m.group(1))
    segs: list[list] = [[first_no, []]]
    for line in lines:
        m = CIRCLED_RE.match(line.text.strip())
        if m and segs[-1][1]:
            segs.append([circled_to_int(m.group(1)), []])
        segs[-1][1].append(line)
    if len(segs) >= 2 and segs[0][0] is None:
        segs = [[segs[1][0], segs[0][1] + segs[1][1]]] + segs[2:]
    return [(no, seg_lines) for no, seg_lines in segs]


def _split_items(lines: list[Line]) -> list[tuple[int | None, list[Line]]]:
    segs: list[list] = [[None, []]]
    last = 0
    for line in lines:
        m = ITEM_RE.match(line.text.strip())
        if m and segs[-1][1]:
            n = int(m.group(1))
            if n == last + 1 or (last == 0 and n == 2):
                segs.append([n, []])
                last = n
        segs[-1][1].append(line)
    if len(segs) >= 2:
        if segs[1][0] == 1:
            segs = [[1, segs[0][1] + segs[1][1]]] + segs[2:]
        else:
            segs[0][0] = 1
    return [(no, seg_lines) for no, seg_lines in segs]


def chunk_units(units: list[Unit], doc_id: str) -> tuple[list[Chunk], list[ExcludedSegment]]:
    chunks: list[Chunk] = []
    excluded: list[ExcludedSegment] = []
    for unit in units:
        if unit.excluded_reason or unit.deleted:
            text = _seg_text(unit.lines)
            excluded.append(
                ExcludedSegment(
                    reason=unit.excluded_reason or "deleted",
                    scope=unit.scope,
                    article_ref=article_ref(unit),
                    page_start=min(x.page for x in unit.lines),
                    page_end=max(x.page_end for x in unit.lines),
                    char_len=len(text),
                    text=text,
                )
            )
            continue
        if len(_seg_text(unit.lines)) <= MAX_CHARS:
            groups: list[tuple[int | None, int | None, list[Line]]] = [(None, None, unit.lines)]
        elif unit.article_num is not None and len(_split_paragraphs(unit.lines)) >= 2:
            groups = []
            for no, seg_lines in _split_paragraphs(unit.lines):
                if len(_seg_text(seg_lines)) > MAX_CHARS:
                    groups.extend((no, None, piece) for piece in _pack(seg_lines))
                else:
                    groups.append((no, None, seg_lines))
        elif unit.article_num is not None and len(_split_items(unit.lines)) >= 2:
            groups = []
            for item_no, seg_lines in _split_items(unit.lines):
                if len(_seg_text(seg_lines)) > MAX_CHARS:
                    groups.extend((None, item_no, piece) for piece in _pack(seg_lines))
                else:
                    groups.append((None, item_no, seg_lines))
        else:
            groups = [(None, None, piece) for piece in _pack(unit.lines)]
        total = len(groups)
        for idx, (no, item_no, glines) in enumerate(groups):
            split_index = idx + 1 if total > 1 else None
            split_total = total if total > 1 else None
            text = _seg_text(glines)
            seq = len(chunks)
            doc_part_title = unit.article_title or unit.appendix_title
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{seq:04d}",
                    doc_id=doc_id,
                    seq=seq,
                    part=unit.part,
                    scope=unit.scope,
                    scope_label=unit.scope_label,
                    chapter=unit.chapter,
                    section=unit.section,
                    article_ref=article_ref(unit),
                    article_no=article_no_of(unit),
                    article_title=unit.article_title,
                    paragraph_no=no,
                    split_index=split_index,
                    split_total=split_total,
                    appendix_label=unit.appendix_label,
                    title=doc_part_title,
                    page_start=min(x.page for x in glines),
                    page_end=max(x.page_end for x in glines),
                    header=make_header(unit, no, split_index, split_total, item_no=item_no),
                    text=text,
                    embed_text="",
                    char_len=len(text),
                )
            )
    for chunk in chunks:
        chunk.embed_text = chunk.header + "\n" + chunk.text
    return chunks, excluded


def chunk_lines(lines: list[Line], doc_id: str) -> tuple[list[Chunk], list[ExcludedSegment], list[Line]]:
    units, structure = build_units(lines)
    chunks, excluded = chunk_units(units, doc_id)
    return chunks, excluded, structure


def write_chunks_json(
    path: Path,
    doc_id: str,
    pdf: str,
    sha256: str,
    chunks: list[Chunk],
    excluded: list[ExcludedSegment],
    structure_lines: list[Line],
) -> None:
    data = {
        "doc_id": doc_id,
        "source_pdf": pdf,
        "sha256": sha256,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chunk_count": len(chunks),
        "chunks": [asdict(c) for c in chunks],
        "excluded": [asdict(e) for e in excluded],
        "structure_lines": [{"page": x.page, "text": x.text} for x in structure_lines],
    }
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
