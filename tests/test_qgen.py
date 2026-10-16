import json

import pytest

from apt_rag.cli import main
from apt_rag.config import Settings
from apt_rag.evaluate import load_questions
from apt_rag.llm import OllamaClient, OllamaConnectionError
from apt_rag.models import Chunk
from apt_rag.qgen import SYSTEM_PROMPT, generate, select_chunks, split_parts
from apt_rag.service import generate_questions


def make_chunk(seq, ref, *, part="본문", chapter="제1장 총칙", char_len=200, page=1, title=None):
    text = "관리규약 본문 내용입니다. " * 10
    title = title or f"{ref} 제목"
    return Chunk(
        chunk_id=f"d:{seq:04d}",
        doc_id="d",
        seq=seq,
        part=part,
        scope=part,
        scope_label="",
        chapter=chapter,
        section="",
        article_ref=ref,
        article_no=ref.replace("제", "").replace("조", ""),
        article_title=title,
        paragraph_no=None,
        split_index=None,
        split_total=None,
        appendix_label="",
        title=title,
        page_start=page,
        page_end=page,
        header=f"[{chapter} > {ref}({title})]",
        text=text,
        embed_text=text,
        char_len=char_len,
    )


class FakeLLM:
    name = "fake"

    def __init__(self, replies=None):
        self.replies = replies or {}
        self.calls = []
        self.counter = 0

    def check_model(self):
        pass

    def complete(self, system, messages, *, temperature=0.0, max_tokens=1024, seed=None, json_mode=False):
        self.calls.append((messages, seed))
        self.counter += 1
        user = messages[0]["content"]
        for ref, reply in self.replies.items():
            if ref in user:
                return reply
        return json.dumps({"questions": [f"입주민 질문 번호 {self.counter}은 어떻게 되나요"]}, ensure_ascii=False)


def sample_chunks():
    chunks = []
    seq = 0
    for chapter_no in range(3):
        for article_no in range(1, 9):
            seq += 1
            chunks.append(
                make_chunk(
                    seq,
                    f"제{chapter_no * 10 + article_no}조",
                    chapter=f"제{chapter_no + 1}장 장{chapter_no}",
                )
            )
    return chunks


def test_selection_is_reproducible_with_same_seed():
    chunks = sample_chunks()
    first = select_chunks(chunks, parts=("본문",), min_chars=80, count=6, seed=42)
    second = select_chunks(chunks, parts=("본문",), min_chars=80, count=6, seed=42)
    assert [c.article_ref for c in first] == [c.article_ref for c in second]
    assert len(first) == 6


def test_selection_spreads_across_chapters():
    chunks = sample_chunks()
    picked = select_chunks(chunks, parts=("본문",), min_chars=80, count=3, seed=7)
    assert len({c.chapter for c in picked}) == 3


def test_parts_and_min_chars_filter():
    chunks = [
        make_chunk(1, "제1조"),
        make_chunk(2, "제2조", char_len=20),
        make_chunk(3, "별표 1", part="별표"),
        make_chunk(4, "제3조"),
    ]
    picked = select_chunks(chunks, parts=("본문",), min_chars=80, count=10, seed=1)
    assert sorted(c.article_ref for c in picked) == ["제1조", "제3조"]
    with_appendix = select_chunks(chunks, parts=split_parts("본문, 별표"), min_chars=80, count=10, seed=1)
    assert "별표 1" in [c.article_ref for c in with_appendix]


def test_one_candidate_per_article_ref():
    chunks = [
        make_chunk(1, "제5조"),
        make_chunk(2, "제5조"),
        make_chunk(3, "제6조"),
    ]
    picked = select_chunks(chunks, parts=("본문",), min_chars=80, count=10, seed=1)
    assert [c.seq for c in picked] == [1, 3]


def test_article_number_question_is_dropped(tmp_path):
    chunks = [make_chunk(1, "제69조", title="관리비등의 연체료")]
    client = FakeLLM(
        {
            "제69조": json.dumps(
                {"questions": ["제69조에 따르면 연체료는 얼마인가요?", "관리비 늦게 내면 이자가 붙나요?"]},
                ensure_ascii=False,
            )
        }
    )
    result = generate(chunks, out=tmp_path / "q.csv", llm=client)
    assert [q.question for q in result.questions] == ["관리비 늦게 내면 이자가 붙나요?"]


def test_duplicate_questions_removed(tmp_path):
    chunks = [make_chunk(1, "제5조"), make_chunk(2, "제6조")]
    same = json.dumps({"questions": ["반려동물을 키워도 되나요?"]}, ensure_ascii=False)
    client = FakeLLM({"제5조": same, "제6조": same})
    result = generate(chunks, out=tmp_path / "q.csv", llm=client)
    assert len(result.questions) == 1


def test_parse_failure_is_skipped(tmp_path):
    chunks = [make_chunk(1, "제5조"), make_chunk(2, "제6조")]
    client = FakeLLM({"제5조": "이건 JSON이 아닙니다"})
    result = generate(chunks, out=tmp_path / "q.csv", llm=client)
    assert result.skipped == ["제5조"]
    assert [q.expected_article for q in result.questions] == ["제6조"]


def test_csv_is_readable_by_load_questions(tmp_path):
    chunks = [make_chunk(1, "제69조", page=12, title="관리비등의 연체료")]
    client = FakeLLM({"제69조": json.dumps({"questions": ["관리비를 늦게 내면 어떻게 되나요?"]}, ensure_ascii=False)})
    out = tmp_path / "q.csv"
    generate(chunks, out=out, llm=client)
    assert load_questions(out) == [("관리비를 늦게 내면 어떻게 되나요?", "제69조")]
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert header == "question,expected_article,source_title,source_pages"


def test_existing_file_refused_then_force_overwrites(tmp_path):
    chunks = [make_chunk(1, "제5조")]
    out = tmp_path / "q.csv"
    out.write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        generate(chunks, out=out, llm=FakeLLM())
    assert out.read_text(encoding="utf-8") == "old"
    generate(chunks, out=out, llm=FakeLLM(), force=True)
    assert out.read_text(encoding="utf-8") != "old"


def test_generate_uses_json_mode_and_seed(tmp_path):
    seen = {}

    class RecordingLLM(FakeLLM):
        def complete(self, system, messages, *, temperature=0.0, max_tokens=1024, seed=None, json_mode=False):
            seen.update(json_mode=json_mode, seed=seed, system=system)
            return super().complete(
                system,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                json_mode=json_mode,
            )

    generate([make_chunk(1, "제5조")], out=tmp_path / "q.csv", llm=RecordingLLM(), seed=7)
    assert seen["json_mode"] is True
    assert seen["seed"] == 7
    assert seen["system"] == SYSTEM_PROMPT


def test_service_requires_index(settings):
    with pytest.raises(FileNotFoundError):
        generate_questions(settings, out=settings.data_dir / "q.csv", llm=FakeLLM())


def test_cli_without_index_exits_one(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APT_RAG_DATA_DIR", str(tmp_path / "data"))
    assert main(["gen-questions", "--out", str(tmp_path / "q.csv")]) == 1
    assert "먼저 index" in capsys.readouterr().err


def test_cli_connection_failure_exits_one(tmp_path, monkeypatch, capsys):
    index_dir = tmp_path / "data" / "index"
    index_dir.mkdir(parents=True)
    rows = [json.loads(json.dumps(c.__dict__)) for c in sample_chunks()]
    (index_dir / "chunks.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APT_RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("APT_RAG_OLLAMA_URL", "http://127.0.0.1:1")
    assert main(["gen-questions", "--out", str(tmp_path / "q.csv")]) == 1
    err = capsys.readouterr().err
    assert "ollama serve" in err and "APT_RAG_OLLAMA_URL" in err
    assert not (tmp_path / "q.csv").exists()


@pytest.mark.slow
def test_real_ollama_generates_questions(tmp_path):
    client = OllamaClient("http://localhost:11434", "qwen2.5:7b")
    try:
        client.check_model()
    except OllamaConnectionError:
        pytest.skip("ollama 서버가 실행 중이지 않습니다")
    result = generate([make_chunk(1, "제69조", title="관리비등의 연체료")], out=tmp_path / "q.csv", llm=client, count=1)
    assert result.selected == 1
