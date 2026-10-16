from __future__ import annotations

from collections import Counter

from apt_rag.ingest.chunk import MAX_CHARS, Unit, article_label, article_no_of, article_ref
from apt_rag.models import (
    Chunk,
    ExcludedSegment,
    ExtractResult,
    Issue,
    Line,
    ValidationReport,
    count_content_chars,
)

SHORT_CHARS = 40
LIST_LIMIT = 50
_SEVERITY_ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2}


def validate(
    extract: ExtractResult,
    page_count: int,
    units: list[Unit],
    chunks: list[Chunk],
    excluded: list[ExcludedSegment],
    structure_lines: list[Line],
) -> ValidationReport:
    issues: list[Issue] = []
    scopes: dict[str, dict] = {}

    by_scope: dict[str, list[Unit]] = {}
    for unit in units:
        if unit.article_num is None or unit.excluded_reason:
            continue
        by_scope.setdefault(unit.scope, []).append(unit)

    if not by_scope.get("본문"):
        issues.append(Issue("ERROR", "NO_ARTICLES", "본문 조 단위가 없습니다.", {}))

    for scope, scope_units in by_scope.items():
        keys = [(u.article_num, u.article_sub or 0) for u in scope_units]
        nums = {u.article_num for u in scope_units}
        top = max(nums)
        missing = [n for n in range(1, top + 1) if n not in nums]
        counts = Counter(keys)
        duplicates = []
        for key, count in counts.items():
            if count >= 2:
                no = article_label(key[0], key[1] or None)
                duplicates.append(no)
                issues.append(
                    Issue(
                        "ERROR",
                        "DUPLICATE_ARTICLE",
                        f"{scope}: {no} 중복 ({count}회)",
                        {"scope": scope, "article": no, "count": count},
                    )
                )
        out_of_order = []
        prev: tuple[int, int] | None = None
        for unit, key in zip(scope_units, keys):
            if prev is not None and key <= prev:
                out_of_order.append(article_no_of(unit))
                issues.append(
                    Issue(
                        "WARN",
                        "OUT_OF_ORDER",
                        f"{scope}: 조 순서 역전 {article_no_of(unit)}",
                        {"scope": scope, "article": article_no_of(unit)},
                    )
                )
            prev = key
        if missing:
            issues.append(
                Issue(
                    "WARN",
                    "MISSING_ARTICLE",
                    f"{scope}: 누락 조 {missing}",
                    {"scope": scope, "missing": missing},
                )
            )
        deleted = [article_ref(u) for u in scope_units if u.deleted]
        scopes[scope] = {
            "articles": sorted({article_no_of(u) for u in scope_units}, key=lambda s: _sort_key(s)),
            "article_units": len(scope_units),
            "missing": missing,
            "duplicates": duplicates,
            "out_of_order": out_of_order,
            "deleted": deleted,
        }

    for chunk in chunks:
        if chunk.char_len > MAX_CHARS:
            issues.append(
                Issue(
                    "WARN",
                    "LONG_CHUNK",
                    f"{chunk.chunk_id} {chunk.article_ref}: {chunk.char_len}자",
                    {"chunk_id": chunk.chunk_id, "article_ref": chunk.article_ref, "char_len": chunk.char_len, "part": chunk.part},
                )
            )
        if chunk.char_len < SHORT_CHARS:
            issues.append(
                Issue(
                    "WARN",
                    "SHORT_CHUNK",
                    f"{chunk.chunk_id} {chunk.article_ref}: {chunk.char_len}자",
                    {"chunk_id": chunk.chunk_id, "article_ref": chunk.article_ref, "char_len": chunk.char_len, "part": chunk.part},
                )
            )

    raw = extract.raw_chars
    extracted = extract.extracted_chars
    extraction_ratio = (extracted + extract.removed_chars) / raw if raw else 1.0
    if extraction_ratio < 0.95:
        issues.append(
            Issue(
                "ERROR",
                "EXTRACTION_LOSS",
                f"추출 누락: 비율 {extraction_ratio:.4f}",
                {"extraction_ratio": extraction_ratio},
            )
        )
    elif extraction_ratio < 0.98 or extraction_ratio > 1.02:
        issues.append(
            Issue(
                "WARN",
                "EXTRACTION_RATIO",
                f"추출 비율 이상: {extraction_ratio:.4f}",
                {"extraction_ratio": extraction_ratio},
            )
        )

    covered_chars = (
        sum(count_content_chars(c.text) for c in chunks)
        + sum(count_content_chars(e.text) for e in excluded)
        + sum(count_content_chars(x.text) for x in structure_lines)
    )
    chunk_ratio = covered_chars / extracted if extracted else 1.0
    if abs(chunk_ratio - 1) > 0.001:
        issues.append(
            Issue(
                "ERROR",
                "CHUNK_LOSS",
                f"청크 누락/중복: 비율 {chunk_ratio:.4f}",
                {"chunk_ratio": chunk_ratio},
            )
        )

    covered_pages: set[int] = set()
    for line in extract.lines:
        covered_pages.update(range(line.page, line.page_end + 1))
    empty_pages = [p for p in range(1, page_count + 1) if p not in covered_pages]
    if empty_pages:
        issues.append(Issue("INFO", "EMPTY_PAGES", f"본문 줄이 없는 쪽: {empty_pages}", {"pages": empty_pages}))

    n_err = sum(1 for i in issues if i.severity == "ERROR")
    n_warn = sum(1 for i in issues if i.severity == "WARN")
    summary = {
        "pages": page_count,
        "chunks_total": len(chunks),
        "chunks_by_part": dict(Counter(c.part for c in chunks)),
        "excluded_by_reason": dict(Counter(e.reason for e in excluded)),
        "extraction_ratio": round(extraction_ratio, 4),
        "chunk_ratio": round(chunk_ratio, 4),
        "errors": n_err,
        "warnings": n_warn,
    }
    return ValidationReport(summary=summary, scopes=scopes, issues=issues, has_errors=n_err > 0)


def _sort_key(article_no: str) -> tuple[int, int]:
    if "의" in article_no:
        num, sub = article_no.split("의", 1)
        return int(num), int(sub)
    return int(article_no), 0


def render_report_md(report: ValidationReport) -> str:
    lines = ["# 검증 리포트", "", "## 요약", "", "| 항목 | 값 |", "|---|---|"]
    for key, value in report.summary.items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## scope별", ""]
    for scope, info in report.scopes.items():
        lines.append(f"### {scope}")
        lines.append(f"- 조 수: {info['article_units']}")
        lines.append(f"- 누락: {info['missing']}")
        lines.append(f"- 중복: {info['duplicates']}")
        lines.append(f"- 순서 역전: {info['out_of_order']}")
        lines.append(f"- 삭제: {info['deleted']}")
        lines.append("")
    lines += ["## 이슈", ""]
    ordered = sorted(report.issues, key=lambda i: _SEVERITY_ORDER.get(i.severity, 9))
    shown: Counter[str] = Counter()
    hidden: Counter[str] = Counter()
    for issue in ordered:
        if issue.code in ("LONG_CHUNK", "SHORT_CHUNK"):
            if shown[issue.code] >= LIST_LIMIT:
                hidden[issue.code] += 1
                continue
            shown[issue.code] += 1
        lines.append(f"- [{issue.severity}] {issue.code}: {issue.message}")
    for code, count in hidden.items():
        lines.append(f"- {code}: 외 {count}건")
    return "\n".join(lines) + "\n"
