import pytest

from apt_rag.evaluate import evaluate_csv, normalize_ref
from apt_rag.search import Searcher
from apt_rag.store import write_index
from test_search import build, make_chunk


def test_normalize_ref_cases():
    assert normalize_ref("제 23 조 제2항") == "제23조"
    assert normalize_ref("별표 8") == "별표8"
    assert normalize_ref("부칙 제1조") == "부칙제1조"


def test_evaluate_hit_and_failure(settings, hash_embedder, tmp_path):
    chunks = [
        make_chunk(0, "제1조", "이 규약은 목적을 정한다."),
        make_chunk(1, "제5조", "입주자는 반려동물을 사육할 수 있으며 사육 기준을 따른다."),
        make_chunk(2, "제6조", "주차장 이용 시간은 별도로 정한다."),
    ]
    build(settings, hash_embedder, chunks)
    csv_path = tmp_path / "q.csv"
    csv_path.write_text(
        "question,expected_article\n"
        "반려동물 사육 기준은?,제5조\n"
        "주차장 이용 시간은?,제1조\n",
        encoding="utf-8",
    )
    result = evaluate_csv(csv_path, Searcher(settings, hash_embedder), k=1)
    assert result.total == 2
    assert result.hit_rate == 0.5
    assert len(result.failures) == 1
    assert result.failures[0].expected == ["제1조"]


def test_missing_column_raises(settings, hash_embedder, tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("question,answer\n질문,제1조\n", encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_csv(csv_path, Searcher(settings, hash_embedder), k=5)
