from apt_rag.pdfinspect import inspect_pdf_file
from pdf_factory import make_pdf


def test_synthetic_kinds(tmp_path):
    pdf = make_pdf(
        tmp_path / "mixed.pdf",
        [
            ["가나다라마바사아자차카타파하 가나다라마바사아자차카타파하 본문입니다"],
            ["이미지로 바뀌는 쪽의 원문 텍스트입니다 충분히 길게"],
            [""],
        ],
        image_pages={1},
    )
    report = inspect_pdf_file(pdf)
    assert [p.kind for p in report.pages] == ["text", "image", "empty"]
    assert report.ocr_pages == [2]


def test_example_pdf_inspect(example_pdf):
    report = inspect_pdf_file(example_pdf)
    assert report.counts.get("image", 0) == 0
    empty = {p.page for p in report.pages if p.kind == "empty"}
    assert {1, 3, 92, 97, 106, 127, 161} <= empty
