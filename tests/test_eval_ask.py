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
