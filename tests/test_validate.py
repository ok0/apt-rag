import dataclasses

from apt_rag.ingest.chunk import build_units, chunk_units
from apt_rag.models import ExtractResult, Line, count_content_chars
from apt_rag.ingest.validate import validate


def L(text, page=1):
    return Line(text=text, page=page, page_end=page)


def pipeline(lines):
    units, structure = build_units(lines)
    chunks, excluded = chunk_units(units, "d")
    total = sum(count_content_chars(x.text) for x in lines)
    extract = ExtractResult(
        doc_id="d",
        lines=lines,
        removed=[],
        raw_chars=total,
        extracted_chars=total,
        removed_chars=0,
        table_count=0,
    )
    return units, chunks, excluded, structure, extract


def codes(report):
    return {i.code for i in report.issues}


def test_missing_article_is_warning():
    lines = [
        L("제1조(목적)"), L("가나다라마바사 본문"),
        L("제2조(정의)"), L("가나다라마바사 본문"),
        L("제4조(범위)"), L("가나다라마바사 본문"),
    ]
    units, chunks, excluded, structure, extract = pipeline(lines)
    report = validate(extract, 1, units, chunks, excluded, structure)
    missing = [i for i in report.issues if i.code == "MISSING_ARTICLE"]
    assert missing and missing[0].severity == "WARN"
    assert missing[0].detail["missing"] == [3]
    assert not report.has_errors


def test_duplicate_article_is_error():
    lines = [
        L("제1조(목적)"), L("가나다라마바사 본문"),
        L("제2조(정의)"), L("가나다라마바사 본문"),
        L("제2조(정의)"), L("가나다라마바사 본문"),
    ]
    units, chunks, excluded, structure, extract = pipeline(lines)
    report = validate(extract, 1, units, chunks, excluded, structure)
    assert "DUPLICATE_ARTICLE" in codes(report)
    assert report.has_errors


def test_chunk_loss_is_error():
    lines = [
        L("제1조(목적)"), L("본문 내용이 충분히 긴 줄입니다 가나다라마바사"),
    ]
    units, chunks, excluded, structure, extract = pipeline(lines)
    chunks = [dataclasses.replace(chunks[0], text="")]
    report = validate(extract, 1, units, chunks, excluded, structure)
    assert "CHUNK_LOSS" in codes(report)
    assert report.has_errors
