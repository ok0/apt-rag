import pytest

from apt_rag.ingest.extract import PAGE_NUM_RE, extract_pdf, table_to_markdown
from apt_rag.ingest.ocr import NullOcr, OcrUnavailableError
from apt_rag.ingest.pdfinspect import inspect_pdf_file
from apt_rag.service import extract
from pdf_factory import make_pdf


def test_reflow_joins_across_pages_and_drops_page_numbers(tmp_path):
    pdf = make_pdf(
        tmp_path / "reflow.pdf",
        [
            ["제1조(목적) 이 규약은 개정 방향과 취지에 ", "적합하도록 공동주택으로 전"],
            ["환하였거나 법 제2", "0조제9항에 따른다.", "", "제2조(정의) 정의는 다음과 같다."],
        ],
    )
    report = inspect_pdf_file(pdf)
    result = extract_pdf(pdf, report, NullOcr())
    texts = [line.text for line in result.lines]
    assert texts[0] == "제1조(목적) 이 규약은 개정 방향과 취지에 적합하도록 공동주택으로 전환하였거나 법 제20조제9항에 따른다."
    assert {(r.page, r.reason) for r in result.removed} == {(1, "page_number"), (2, "page_number")}


def test_table_page_produces_markdown(tmp_path):
    pdf = make_pdf(
        tmp_path / "table.pdf",
        [["표 앞의 본문 문장입니다 충분히 길게 작성합니다"]],
        tables={0: [["비목", "산정방법"], ["일반관리비", "월간 실제 소요"]]},
    )
    report = inspect_pdf_file(pdf)
    result = extract_pdf(pdf, report, NullOcr())
    table_lines = [line for line in result.lines if line.kind == "table"]
    assert any("| 비목 | 산정방법 |" in line.text for line in table_lines)


def test_image_page_without_ocr_raises(tmp_path, settings):
    pdf = make_pdf(
        tmp_path / "image.pdf",
        [["스캔 이미지 쪽 원문 텍스트 여러 글자 입니다 충분히 길게"]],
        image_pages={0},
    )
    with pytest.raises(OcrUnavailableError):
        extract(pdf, settings)


def test_table_to_markdown_vertical_cells():
    md = table_to_markdown([["일\n반\n관\n리\n비", "월간 실제\n소요"]])
    lines = md.split("\n")
    assert lines[0] == "| 일반관리비 | 월간 실제 소요 |"
    assert lines[1] == "|---|---|"


def test_example_pdf_extraction(example_pdf):
    report = inspect_pdf_file(example_pdf)
    result = extract_pdf(example_pdf, report, NullOcr())
    assert not any(PAGE_NUM_RE.match(line.text) for line in result.lines if line.kind == "text")
    joined = next(line for line in result.lines if "공동주택으로 전환하였거나" in line.text)
    assert joined.page == 16
    assert joined.page_end == 17
    assert any("법 제20조제9항" in line.text for line in result.lines)
    ratio = (result.extracted_chars + result.removed_chars) / result.raw_chars
    assert 0.98 <= ratio <= 1.02
