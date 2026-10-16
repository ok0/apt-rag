import pytest

import apt_rag.cli as cli
from apt_rag.generation.answer import DISCLAIMER, NO_HIT_ANSWER
from apt_rag.cli import main
from apt_rag.generation.llm import OllamaConnectionError
from apt_rag.models import AnswerResult, ChatTurn, SearchHit
from apt_rag.retrieval.store import IndexMismatchError


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_help_command_lists_commands(capsys):
    assert main(["help"]) == 0
    out = capsys.readouterr().out
    assert "inspect" in out and "search" in out and "help" in out and "ask" in out


def test_help_command_for_subcommand(capsys):
    assert main(["help", "search"]) == 0
    assert "--top-n" in capsys.readouterr().out


def test_help_command_unknown_topic(capsys):
    assert main(["help", "nope"]) == 1
    assert "nope" in capsys.readouterr().err


def test_search_no_hits_prints_message(monkeypatch, capsys):
    class Empty:
        def __init__(self, settings):
            pass

        def search(self, query, top_n=5, min_sim=None):
            return []

    monkeypatch.setattr(cli, "Searcher", Empty)
    monkeypatch.setattr(cli, "load_settings", lambda: None)
    assert main(["search", "오늘 점심"]) == 0
    assert "관련 조항을 찾지 못했습니다" in capsys.readouterr().out


def test_search_index_mismatch_exits_one(monkeypatch, capsys):
    class Mismatched:
        def __init__(self, settings):
            pass

        def search(self, query, top_n=5, min_sim=None):
            raise IndexMismatchError("색인 불일치: index를 다시 실행하세요")

    monkeypatch.setattr(cli, "Searcher", Mismatched)
    monkeypatch.setattr(cli, "load_settings", lambda: None)
    assert main(["search", "q"]) == 1
    assert "색인 불일치" in capsys.readouterr().err


def test_search_prints_vec_sim_and_passes_min_sim(monkeypatch, capsys):
    seen = {}

    class Fake:
        def __init__(self, settings):
            pass

        def search(self, query, top_n=5, min_sim=None):
            seen["min_sim"] = min_sim
            return [
                SearchHit(
                    rank=1,
                    article_ref="제5조",
                    part="본문",
                    title="반려동물 사육",
                    header="[제5조]",
                    chapter="",
                    page_start=1,
                    page_end=1,
                    score=0.03,
                    vec_rank=1,
                    bm25_rank=None,
                    vec_sim=0.62,
                    chunk_id="d:0001",
                    text="사육 기준",
                )
            ]

    monkeypatch.setattr(cli, "Searcher", Fake)
    monkeypatch.setattr(cli, "load_settings", lambda: None)
    assert main(["search", "반려동물", "--min-sim", "0.3"]) == 0
    assert seen["min_sim"] == 0.3
    assert "vec #1 sim=0.62, bm25 #-" in capsys.readouterr().out


def test_search_marks_band_hits(monkeypatch, capsys):
    class Fake:
        def __init__(self, settings):
            pass

        def search(self, query, top_n=5, min_sim=None):
            return [
                SearchHit(
                    rank=1,
                    article_ref="제93조",
                    part="본문",
                    title="발코니 관리",
                    header="[제93조]",
                    chapter="",
                    page_start=1,
                    page_end=1,
                    score=0.03,
                    vec_rank=1,
                    bm25_rank=2,
                    vec_sim=0.54,
                    chunk_id="d:0004",
                    text="발코니 관리 기준",
                    via_band=True,
                )
            ]

    monkeypatch.setattr(cli, "Searcher", Fake)
    monkeypatch.setattr(cli, "load_settings", lambda: None)
    assert main(["search", "베란다 수선"]) == 0
    out = capsys.readouterr().out
    assert "벡터와 BM25가 모두 상위" in out
    assert "vec #1 sim=0.54 구간통과" in out


def test_band_options_override_settings(monkeypatch, settings):
    seen = {}

    class Fake:
        def __init__(self, settings):
            seen["sim_band"] = settings.sim_band
            seen["band_top_k"] = settings.band_top_k

        def search(self, query, top_n=5, min_sim=None):
            return []

    monkeypatch.setattr(cli, "Searcher", Fake)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    assert main(["search", "q", "--sim-band", "0.03", "--band-top-k", "7"]) == 0
    assert seen == {"sim_band": 0.03, "band_top_k": 7}


def _ask_hit():
    return SearchHit(
        rank=1,
        article_ref="제5조",
        part="본문",
        title="반려동물 사육",
        header="[제5조]",
        chapter="",
        page_start=1,
        page_end=1,
        score=0.03,
        vec_rank=1,
        bm25_rank=None,
        vec_sim=0.62,
        chunk_id="d:0001",
        text="사육 기준",
    )


def test_ask_prints_answer_hits_warning_disclaimer(monkeypatch, capsys, settings):
    seen = {}

    def fake_answer(question, settings, **kwargs):
        seen.update(kwargs)
        return AnswerResult(
            question="q",
            search_query="q",
            rewritten_question=None,
            answer="가능합니다 (제5조)",
            disclaimer=DISCLAIMER,
            hits=[_ask_hit()],
            cited_refs=["제5조"],
            unverified_refs=["제99조"],
            llm="fake",
        )

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "answer", fake_answer)
    assert main(["ask", "q"]) == 0
    out = capsys.readouterr().out
    assert "가능합니다 (제5조)" in out
    assert "제5조 반려동물 사육 (p.1) [인용]" in out
    assert "제99조" in out
    assert "(LLM: fake)" in out
    assert out.strip().splitlines()[-1] == DISCLAIMER


def test_ask_no_hits(monkeypatch, capsys, settings):
    def fake_answer(question, settings, **kwargs):
        return AnswerResult(
            question="q",
            search_query="q",
            rewritten_question=None,
            answer=NO_HIT_ANSWER,
            disclaimer=DISCLAIMER,
            hits=[],
            cited_refs=[],
            unverified_refs=[],
            llm=None,
        )

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "answer", fake_answer)
    assert main(["ask", "오늘 점심"]) == 0
    out = capsys.readouterr().out
    assert NO_HIT_ANSWER in out
    assert DISCLAIMER in out
    assert "LLM:" not in out


def test_ask_bad_history_exits_one(monkeypatch, capsys, settings, tmp_path):
    bad = tmp_path / "history.json"
    bad.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    assert main(["ask", "q", "--history", str(bad)]) == 1
    assert "대화 이력" in capsys.readouterr().err


def test_ask_llm_error_exits_one(monkeypatch, capsys, settings):
    def fake_answer(question, settings, **kwargs):
        raise OllamaConnectionError()

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "answer", fake_answer)
    assert main(["ask", "q"]) == 1
    assert "ollama serve" in capsys.readouterr().err


def test_ask_history_passed(monkeypatch, capsys, settings, tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        '[{"role": "user", "content": "반려동물 키워도 되나요"}, {"role": "assistant", "content": "가능합니다"}]',
        encoding="utf-8",
    )
    seen = {}

    def fake_answer(question, settings, **kwargs):
        seen["history"] = kwargs["history"]
        return AnswerResult(
            question="q",
            search_query="q",
            rewritten_question=None,
            answer=NO_HIT_ANSWER,
            disclaimer=DISCLAIMER,
            hits=[],
            cited_refs=[],
            unverified_refs=[],
            llm=None,
        )

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "answer", fake_answer)
    assert main(["ask", "q", "--history", str(path)]) == 0
    assert len(seen["history"]) == 2
    assert all(isinstance(turn, ChatTurn) for turn in seen["history"])


def test_ask_prints_original_and_rewritten_question(monkeypatch, capsys, settings, tmp_path):
    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    def fake_answer(question, settings, **kwargs):
        return AnswerResult(
            question=question,
            search_query="반려동물은 몇 마리까지 키울 수 있나요?",
            rewritten_question="반려동물은 몇 마리까지 키울 수 있나요?",
            answer="답",
            disclaimer=DISCLAIMER,
            hits=[],
            cited_refs=[],
            unverified_refs=[],
            llm="fake",
        )

    monkeypatch.setattr(cli, "answer", fake_answer)
    assert cli.main(["ask", "그럼 몇 마리까지 돼요?"]) == 0
    out = capsys.readouterr().out
    assert "[질문 원본] 그럼 몇 마리까지 돼요?" in out
    assert "[LLM 보정 질문] 반려동물은 몇 마리까지 키울 수 있나요?" in out
