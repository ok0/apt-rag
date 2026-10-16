import dataclasses
import json

import pytest

from apt_rag.models import SearchHit
from apt_rag.service import ASK_EVAL_CATEGORIES, evaluate_answers


def make_hit(rank, ref, text="본문"):
    return SearchHit(
        rank=rank,
        article_ref=ref,
        part="본문",
        title="제목",
        header=f"[{ref}]",
        chapter="제5장",
        page_start=1,
        page_end=1,
        score=0.03,
        vec_rank=1,
        bm25_rank=1,
        vec_sim=0.7,
        chunk_id=f"d:{rank:04d}",
        text=text,
    )


class FakeLLM:
    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, messages, *, temperature=0.0, max_tokens=1024):
        self.calls.append({"system": system, "messages": messages})
        return self.replies.pop(0)


class FakeSearcher:
    def __init__(self, hits_by_query):
        self.hits_by_query = hits_by_query
        self.queries = []

    def search(self, query, top_n=5, min_sim=None):
        self.queries.append(query)
        return self.hits_by_query.get(query, [])[:top_n]

    def expand_context(self, hits):
        return list(hits)


def write_csv(path):
    path.write_text(
        "question,expected_article\n"
        "반려견 목줄은 필요한가요,제23조\n"
        "엘리베이터 점검 주기는 어떻게 되나요,제5조\n"
        "우주선 발사 비용은 얼마인가요,\n",
        encoding="utf-8",
    )
    return path


def test_evaluate_answers_classifies_rows(settings, tmp_path):
    csv_path = write_csv(tmp_path / "ask.csv")
    searcher = FakeSearcher(
        {
            "반려견 목줄은 필요한가요": [make_hit(1, "제23조")],
            "엘리베이터 점검 주기는 어떻게 되나요": [make_hit(1, "제5조")],
        }
    )
    llm = FakeLLM(["목줄을 해야 합니다 (제23조)", "점검 주기는 조항에 있습니다"])
    result = evaluate_answers(csv_path, settings, k=5, searcher=searcher, llm=llm)

    assert result.k == 5
    assert result.total == 3
    assert [row.category for row in result.rows] == [
        "cited",
        "retrieved_not_cited",
        "unanswerable_refused",
    ]
    assert result.rows[2].refused is True
    assert result.rows[2].gold_rank is None
    assert result.rows[0].gold_rank == 1
    assert list(result.counts) == list(ASK_EVAL_CATEGORIES)
    assert result.counts["cited"] == 1
    assert result.counts["retrieved_not_cited"] == 1
    assert result.counts["unanswerable_refused"] == 1
    assert sum(result.counts.values()) == 3
    assert list((settings.work_dir / "eval").glob("ask-eval-*.json"))


def test_evaluate_answers_reports_each_row(settings, tmp_path):
    csv_path = write_csv(tmp_path / "ask.csv")
    searcher = FakeSearcher(
        {
            "반려견 목줄은 필요한가요": [make_hit(1, "제23조")],
            "엘리베이터 점검 주기는 어떻게 되나요": [make_hit(1, "제5조")],
        }
    )
    llm = FakeLLM(["목줄을 해야 합니다 (제23조)", "점검 주기는 조항에 있습니다"])
    seen = []

    def on_row(index, total, row):
        seen.append((index, total, row))

    result = evaluate_answers(csv_path, settings, k=5, searcher=searcher, llm=llm, on_row=on_row)

    assert [(i, total) for i, total, _ in seen] == [(1, 3), (2, 3), (3, 3)]
    assert [row for _, _, row in seen] == result.rows


def test_evaluate_answers_removes_partial_on_success(settings, tmp_path):
    csv_path = write_csv(tmp_path / "ask.csv")
    searcher = FakeSearcher(
        {
            "반려견 목줄은 필요한가요": [make_hit(1, "제23조")],
            "엘리베이터 점검 주기는 어떻게 되나요": [make_hit(1, "제5조")],
        }
    )
    llm = FakeLLM(["목줄을 해야 합니다 (제23조)", "점검 주기는 조항에 있습니다"])
    evaluate_answers(csv_path, settings, k=5, searcher=searcher, llm=llm)

    eval_dir = settings.work_dir / "eval"
    assert not list(eval_dir.glob("ask-eval-*.partial.jsonl"))
    assert list(eval_dir.glob("ask-eval-*.json"))


class FailingSecondLLM(FakeLLM):
    def complete(self, system, messages, *, temperature=0.0, max_tokens=1024):
        if len(self.calls) >= 1:
            raise RuntimeError("boom")
        return super().complete(system, messages, temperature=temperature, max_tokens=max_tokens)


def test_evaluate_answers_keeps_partial_on_failure(settings, tmp_path):
    csv_path = write_csv(tmp_path / "ask.csv")
    searcher = FakeSearcher(
        {
            "반려견 목줄은 필요한가요": [make_hit(1, "제23조")],
            "엘리베이터 점검 주기는 어떻게 되나요": [make_hit(1, "제5조")],
        }
    )
    llm = FailingSecondLLM(["목줄을 해야 합니다 (제23조)"])

    with pytest.raises(RuntimeError):
        evaluate_answers(csv_path, settings, k=5, searcher=searcher, llm=llm)

    partials = list((settings.work_dir / "eval").glob("ask-eval-*.partial.jsonl"))
    assert len(partials) == 1
    lines = partials[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["question"] == "반려견 목줄은 필요한가요"


def test_evaluate_answers_names_result_file_with_model(settings, tmp_path):
    csv_path = write_csv(tmp_path / "ask.csv")
    searcher = FakeSearcher(
        {
            "반려견 목줄은 필요한가요": [make_hit(1, "제23조")],
            "엘리베이터 점검 주기는 어떻게 되나요": [make_hit(1, "제5조")],
        }
    )
    llm = FakeLLM(["목줄을 해야 합니다 (제23조)", "점검 주기는 조항에 있습니다"])
    model_settings = dataclasses.replace(settings, llm_model="hf.co/foo:bar")
    result = evaluate_answers(csv_path, model_settings, k=5, searcher=searcher, llm=llm)

    final_files = list((settings.work_dir / "eval").glob("ask-eval-*.json"))
    assert len(final_files) == 1
    name = final_files[0].name
    assert name.startswith("ask-eval-hf.co_foo_bar-")
    assert ":" not in name and "/" not in name
    assert json.loads(final_files[0].read_text(encoding="utf-8"))["model"] == "hf.co/foo:bar"
    assert result.model == "hf.co/foo:bar"
