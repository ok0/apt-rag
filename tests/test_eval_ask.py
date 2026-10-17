import dataclasses
import json

import pytest

from apt_rag.generation.answer import NOT_FOUND_ANSWER
from apt_rag.models import AskEvalRow, SearchHit
from apt_rag.service import ASK_EVAL_CATEGORIES, ASK_EVAL_OUTCOMES, classify_outcome, evaluate_answers, summarize_ask_eval


def make_hit(rank, ref, text="본문 조항의 내용을 여기에 적습니다"):
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


def reply(answer, evidence="본문 조항의 내용을 여기에 적습니다", answerable=True):
    return json.dumps({"answerable": answerable, "evidence": evidence, "answer": answer}, ensure_ascii=False)


class FakeLLM:
    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, messages, **kwargs):
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
    llm = FakeLLM([reply("목줄을 해야 합니다 (제23조)"), reply(NOT_FOUND_ANSWER, evidence="", answerable=False)])
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
    assert [row.outcome for row in result.rows] == ["correct_cited", "false_refusal", "correct_refusal"]
    assert result.summary["answerable_total"] == 2
    assert result.summary["correct_cited_rate"] == 0.5
    assert result.summary["unanswerable_outcomes"]["correct_refusal"] == 1
    saved = json.loads(list((settings.work_dir / "eval").glob("ask-eval-*.json"))[0].read_text(encoding="utf-8"))
    assert saved["rows"][1]["outcome"] == "false_refusal"
    assert saved["summary"]["answerable_outcomes"]["false_refusal"] == 1


def test_evaluate_answers_mixed_refusal_is_answered(settings, tmp_path):
    csv_path = write_csv(tmp_path / "ask.csv")
    searcher = FakeSearcher(
        {
            "반려견 목줄은 필요한가요": [make_hit(1, "제23조")],
            "엘리베이터 점검 주기는 어떻게 되나요": [make_hit(1, "제5조")],
            "우주선 발사 비용은 얼마인가요": [make_hit(1, "제5조")],
        }
    )
    llm = FakeLLM(
        [
            reply("목줄을 해야 합니다 (제23조)"),
            reply(NOT_FOUND_ANSWER, evidence="", answerable=False),
            reply("관리사무소가 택배 분실을 배상해 주지 않습니다. (관리규약에서 해당 내용을 찾지 못했습니다. 관리사무소에 문의해 주세요.)"),
        ]
    )
    result = evaluate_answers(csv_path, settings, k=5, searcher=searcher, llm=llm)

    assert result.rows[2].category == "unanswerable_answered"
    assert result.rows[2].refused is False


def test_evaluate_answers_pure_refusal_stays_refused(settings, tmp_path):
    csv_path = write_csv(tmp_path / "ask.csv")
    searcher = FakeSearcher(
        {
            "반려견 목줄은 필요한가요": [make_hit(1, "제23조")],
            "엘리베이터 점검 주기는 어떻게 되나요": [make_hit(1, "제5조")],
            "우주선 발사 비용은 얼마인가요": [make_hit(1, "제5조")],
        }
    )
    llm = FakeLLM(
        [
            reply("목줄을 해야 합니다 (제23조)"),
            reply(NOT_FOUND_ANSWER, evidence="", answerable=False),
            reply("관리규약에서 해당 내용을 찾지 못했습니다. 관리사무소에 문의해 주세요."),
        ]
    )
    result = evaluate_answers(csv_path, settings, k=5, searcher=searcher, llm=llm)

    assert result.rows[2].category == "unanswerable_refused"
    assert result.rows[2].refused is True


def test_evaluate_answers_refusal_with_citation_tail_stays_refused(settings, tmp_path):
    csv_path = write_csv(tmp_path / "ask.csv")
    searcher = FakeSearcher(
        {
            "반려견 목줄은 필요한가요": [make_hit(1, "제23조")],
            "엘리베이터 점검 주기는 어떻게 되나요": [make_hit(1, "제5조")],
            "우주선 발사 비용은 얼마인가요": [make_hit(1, "제5조")],
        }
    )
    llm = FakeLLM(
        [
            reply("목줄을 해야 합니다 (제23조)"),
            reply(NOT_FOUND_ANSWER, evidence="", answerable=False),
            reply("관리규약에서 해당 내용을 찾지 못했습니다. 관리사무소에 문의해 주세요. (제85조), (제86조), (별표 3)"),
        ]
    )
    result = evaluate_answers(csv_path, settings, k=5, searcher=searcher, llm=llm)

    assert result.rows[2].category == "unanswerable_refused"
    assert result.rows[2].refused is True


def test_evaluate_answers_reports_each_row(settings, tmp_path):
    csv_path = write_csv(tmp_path / "ask.csv")
    searcher = FakeSearcher(
        {
            "반려견 목줄은 필요한가요": [make_hit(1, "제23조")],
            "엘리베이터 점검 주기는 어떻게 되나요": [make_hit(1, "제5조")],
        }
    )
    llm = FakeLLM([reply("목줄을 해야 합니다 (제23조)"), reply(NOT_FOUND_ANSWER, evidence="", answerable=False)])
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
    llm = FakeLLM([reply("목줄을 해야 합니다 (제23조)"), reply(NOT_FOUND_ANSWER, evidence="", answerable=False)])
    evaluate_answers(csv_path, settings, k=5, searcher=searcher, llm=llm)

    eval_dir = settings.work_dir / "eval"
    assert not list(eval_dir.glob("ask-eval-*.partial.jsonl"))
    assert list(eval_dir.glob("ask-eval-*.json"))


class FailingSecondLLM(FakeLLM):
    def complete(self, system, messages, **kwargs):
        if len(self.calls) >= 1:
            raise RuntimeError("boom")
        return super().complete(system, messages, **kwargs)


def test_evaluate_answers_keeps_partial_on_failure(settings, tmp_path):
    csv_path = write_csv(tmp_path / "ask.csv")
    searcher = FakeSearcher(
        {
            "반려견 목줄은 필요한가요": [make_hit(1, "제23조")],
            "엘리베이터 점검 주기는 어떻게 되나요": [make_hit(1, "제5조")],
        }
    )
    llm = FailingSecondLLM([reply("목줄을 해야 합니다 (제23조)")])

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
    llm = FakeLLM([reply("목줄을 해야 합니다 (제23조)"), reply(NOT_FOUND_ANSWER, evidence="", answerable=False)])
    model_settings = dataclasses.replace(settings, llm_model="hf.co/foo:bar")
    result = evaluate_answers(csv_path, model_settings, k=5, searcher=searcher, llm=llm)

    final_files = list((settings.work_dir / "eval").glob("ask-eval-*.json"))
    assert len(final_files) == 1
    name = final_files[0].name
    assert name.startswith("ask-eval-hf.co_foo_bar-")
    assert ":" not in name and "/" not in name
    assert json.loads(final_files[0].read_text(encoding="utf-8"))["model"] == "hf.co/foo:bar"
    assert result.model == "hf.co/foo:bar"


def test_evaluate_answers_evidence_mismatch_twice_sets_fallback(settings, tmp_path):
    csv_path = write_csv(tmp_path / "ask.csv")
    searcher = FakeSearcher(
        {
            "반려견 목줄은 필요한가요": [make_hit(1, "제23조")],
            "엘리베이터 점검 주기는 어떻게 되나요": [make_hit(1, "제5조")],
            "우주선 발사 비용은 얼마인가요": [make_hit(1, "제5조")],
        }
    )
    bad = reply("우주선 발사 비용은 (제5조)", evidence="완전히 다른 문장입니다 본문에 없습니다")
    llm = FakeLLM(
        [
            reply("목줄을 해야 합니다 (제23조)"),
            reply(NOT_FOUND_ANSWER, evidence="", answerable=False),
            bad,
            bad,
        ]
    )
    result = evaluate_answers(csv_path, settings, k=5, searcher=searcher, llm=llm)

    assert result.rows[2].fallback == "evidence"
    assert result.rows[2].category == "unanswerable_refused"
    assert result.rows[0].fallback is None
    assert [a["result"] for a in result.rows[2].attempts] == ["evidence", "evidence"]


def test_evaluate_answers_row_records_answerable_and_evidence(settings, tmp_path):
    csv_path = write_csv(tmp_path / "ask.csv")
    searcher = FakeSearcher(
        {
            "반려견 목줄은 필요한가요": [make_hit(1, "제23조")],
            "엘리베이터 점검 주기는 어떻게 되나요": [make_hit(1, "제5조")],
        }
    )
    llm = FakeLLM([reply("목줄을 해야 합니다 (제23조)"), reply(NOT_FOUND_ANSWER, evidence="", answerable=False)])
    result = evaluate_answers(csv_path, settings, k=5, searcher=searcher, llm=llm)

    assert result.rows[0].answerable is True
    assert result.rows[0].evidence == "본문 조항의 내용을 여기에 적습니다"
    assert result.rows[1].answerable is False
    assert result.rows[1].evidence == ""

    saved = json.loads(list((settings.work_dir / "eval").glob("ask-eval-*.json"))[0].read_text(encoding="utf-8"))
    assert saved["rows"][0]["answerable"] is True
    assert saved["rows"][0]["evidence"] == "본문 조항의 내용을 여기에 적습니다"
    assert saved["rows"][1]["answerable"] is False
    assert saved["rows"][1]["evidence"] == ""


def _row(**kw):
    base = dict(question="q", expected=["제23조"], category="retrieved_not_cited", gold_rank=1,
                hits=["제23조"], cited_refs=[], unverified_refs=[], refused=False, answer="", answerable=True)
    base.update(kw)
    return AskEvalRow(**base)


@pytest.mark.parametrize(
    "kw,expected_outcome",
    [
        (dict(expected=[], category="unanswerable_refused", refused=True), "correct_refusal"),
        (dict(expected=[], category="unanswerable_answered", refused=False), "answered_unanswerable"),
        (dict(category="cited", cited_refs=["제23조"]), "correct_cited"),
        (dict(category="rank_beyond_k", hits=["제5조"], answerable=False, refused=True), "search_miss"),
        (dict(category="cut_by_min_sim", hits=[]), "search_miss"),
        (dict(category="not_in_candidates", hits=[]), "search_miss"),
        (dict(fallback="evidence", refused=True), "fallback"),
        (
            dict(
                answerable=False,
                refused=True,
                attempts=[{"result": "evidence", "evidence": "x"}, {"result": "unanswerable", "evidence": ""}],
            ),
            "false_refusal",
        ),
        (dict(answerable=True, refused=True), "false_refusal"),
        (dict(cited_refs=["제5조"], hits=["제23조", "제5조"]), "wrong_citation"),
        (dict(cited_refs=[]), "uncited_answer"),
    ],
)
def test_classify_outcome(kw, expected_outcome):
    assert classify_outcome(_row(**kw)) == expected_outcome


def test_summarize_ask_eval_splits_groups():
    rows = [
        _row(category="cited", cited_refs=["제23조"]),
        _row(answerable=False, refused=True),
        _row(category="not_in_candidates", hits=[], refused=True),
        _row(expected=[], category="unanswerable_refused", refused=True),
        _row(expected=[], category="unanswerable_answered"),
    ]
    for o in rows:
        o.outcome = classify_outcome(o)
    s = summarize_ask_eval(rows)
    assert s["answerable_total"] == 3
    assert s["answerable_outcomes"] == {
        "correct_cited": 1,
        "false_refusal": 1,
        "wrong_citation": 0,
        "fallback": 0,
        "uncited_answer": 0,
        "search_miss": 1,
    }
    assert s["correct_cited_rate"] == 0.3333
    assert s["unanswerable_total"] == 2
    assert s["correct_refusal_rate"] == 0.5
    assert s["answered_unanswerable_rate"] == 0.5
    assert s["refused"] == 3
    assert s["refusal_rate"] == 0.6
    assert s["answerable_refused"] == 2
    assert s["answerable_refusal_rate"] == 0.6667


def test_summarize_ask_eval_empty_group_rate_is_none():
    rows = [_row(category="cited", cited_refs=["제23조"])]
    rows[0].outcome = classify_outcome(rows[0])
    s = summarize_ask_eval(rows)
    assert s["unanswerable_total"] == 0
    assert s["correct_refusal_rate"] is None
    assert s["correct_cited_rate"] == 1.0


def test_evaluate_answers_refusal_after_retry_is_false_refusal(settings, tmp_path):
    csv_path = tmp_path / "one.csv"
    csv_path.write_text("question,expected_article\n경비원에게 폭언해도 되나요,제23조\n", encoding="utf-8")
    searcher = FakeSearcher({"경비원에게 폭언해도 되나요": [make_hit(1, "제22조"), make_hit(2, "제23조")]})
    llm = FakeLLM(
        [
            reply("안 됩니다 (제22조)", evidence="완전히 다른 문장입니다 본문에 없습니다"),
            reply(NOT_FOUND_ANSWER, evidence="", answerable=False),
        ]
    )
    result = evaluate_answers(csv_path, settings, k=5, searcher=searcher, llm=llm)

    row = result.rows[0]
    assert row.category == "retrieved_not_cited"
    assert row.outcome == "false_refusal"
    assert [a["result"] for a in row.attempts] == ["evidence", "unanswerable"]
    assert row.gold_rank == 2
    assert result.summary["answerable_outcomes"]["false_refusal"] == 1
