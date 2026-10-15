import pytest

from apt_rag.config import Settings
from apt_rag.service import prepare


@pytest.fixture
def prepared(example_pdf, tmp_path):
    settings = Settings(
        data_dir=tmp_path / "data",
        embed_model="fake-hash",
        device="cpu",
        batch_size=8,
        max_seq_length=512,
        ocr="none",
    )
    return prepare(example_pdf, settings)


def _refs(chunks):
    return {c.article_ref for c in chunks}


def test_example_body_structure(prepared):
    report = prepared.report
    assert report.scopes["본문"]["missing"] == []
    assert report.scopes["본문"]["duplicates"] == []
    body = [c for c in prepared.chunks if c.part == "본문"]
    assert max(int(c.article_no.split("의")[0]) for c in body if c.article_no) == 108
    refs = _refs(prepared.chunks)
    assert {
        "제13조의2",
        "제32조의2",
        "제32조의3",
        "제43조의2",
        "제71조의2",
        "제71조의3",
        "제84조의2",
        "제89조의2",
        "제89조의3",
        "제90조의2",
        "제90조의3",
    } <= refs
    assert all(c.page_end <= 85 for c in body)


def test_example_appendices(prepared):
    refs = _refs(prepared.chunks)
    assert "부칙 제1조" in refs
    assert {f"별표 {n}" for n in range(1, 11)} <= refs
    assert {"별첨 1 제18조", "별첨 2 제16조", "별첨 4 제12조"} <= refs


def test_example_known_content(prepared):
    art7 = [c for c in prepared.chunks if c.article_ref == "제7조"]
    assert any("전환하였거나" in c.text for c in art7)
    art100 = [c for c in prepared.chunks if c.article_ref == "제100조"]
    assert any("제20조제9항" in c.text for c in art100)
    assert len([c for c in prepared.chunks if c.article_ref == "제3조"]) >= 2
    assert len([c for c in prepared.chunks if c.article_ref == "제83조"]) >= 2


def test_example_length_and_coverage(prepared):
    for c in prepared.chunks:
        if c.part not in ("별표", "별지") and c.scope != "별첨서식" and not c.text.startswith("|"):
            assert c.char_len <= 1500, c.chunk_id
    assert not prepared.report.has_errors
    assert abs(prepared.report.summary["chunk_ratio"] - 1) <= 0.001
