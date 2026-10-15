import pytest

import apt_rag.cli as cli
from apt_rag.cli import main
from apt_rag.models import SearchHit


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_help_command_lists_commands(capsys):
    assert main(["help"]) == 0
    out = capsys.readouterr().out
    assert "inspect" in out and "search" in out and "help" in out


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
