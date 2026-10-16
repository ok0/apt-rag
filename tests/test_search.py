import dataclasses

import pytest

from apt_rag.retrieval.bm25 import tokenize
from apt_rag.config import Settings
from apt_rag.retrieval.embed import BgeM3Embedder
from apt_rag.models import Chunk
from apt_rag.ingest.ocr import OcrUnavailableError, get_ocr_engine
from apt_rag.retrieval.search import Searcher, rrf
from apt_rag.retrieval.store import IndexMismatchError, load_index, write_index
from apt_rag.retrieval.synonyms import Synonyms


def make_chunk(seq, ref, text, paragraph_no=None):
    header = f"[{ref}]"
    return Chunk(
        chunk_id=f"d:{seq:04d}",
        doc_id="d",
        seq=seq,
        part="본문",
        scope="본문",
        scope_label="",
        chapter="",
        section="",
        article_ref=ref,
        article_no=ref.replace("제", "").replace("조", ""),
        article_title="",
        paragraph_no=paragraph_no,
        split_index=None,
        split_total=None,
        appendix_label="",
        title="",
        page_start=1,
        page_end=1,
        header=header,
        text=text,
        embed_text=f"{header}\n{text}",
        char_len=len(text),
    )


def manifest_for(embedder, n):
    return {
        "doc_id": "d",
        "source_pdf": "x.pdf",
        "sha256": "x",
        "embed_model": embedder.model_name,
        "dim": len(embedder.embed_query("x")),
        "chunk_count": n,
        "created_at": "2026-01-01T00:00:00+00:00",
    }


def build(settings, embedder, chunks):
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    write_index(settings, chunks, vectors, manifest_for(embedder, len(chunks)))


def test_rrf_prefers_items_in_both_lists():
    score = rrf([["a", "b"], ["b", "c"]])
    assert max(score, key=score.get) == "b"
    assert score["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_search_returns_expected_article(settings, hash_embedder):
    chunks = [
        make_chunk(0, "제1조", "이 규약은 목적을 정한다."),
        make_chunk(1, "제5조", "입주자는 반려동물을 사육할 수 있으며 사육 기준을 따른다."),
        make_chunk(2, "제6조", "주차장 이용 시간은 별도로 정한다."),
    ]
    build(settings, hash_embedder, chunks)
    hits = Searcher(settings, hash_embedder).search("반려동물 사육", 5, min_sim=0)
    assert hits[0].article_ref == "제5조"


def test_same_article_listed_once(settings, hash_embedder):
    chunks = [
        make_chunk(0, "제1조", "이 규약은 목적을 정한다."),
        make_chunk(1, "제5조", "반려동물 사육 기준 ①", paragraph_no=1),
        make_chunk(2, "제5조", "반려동물 사육 허가 ②", paragraph_no=2),
        make_chunk(3, "제6조", "주차장 이용 시간은 별도로 정한다."),
    ]
    build(settings, hash_embedder, chunks)
    hits = Searcher(settings, hash_embedder).search("반려동물 사육", 5)
    refs = [h.article_ref for h in hits]
    assert refs.count("제5조") == 1
    assert refs[0] == "제5조"


def test_model_mismatch_raises(settings, hash_embedder):
    chunks = [make_chunk(0, "제1조", "이 규약은 목적을 정한다.")]
    build(settings, hash_embedder, chunks)
    other = type(hash_embedder)()
    other.model_name = "other-model"
    with pytest.raises(IndexMismatchError):
        Searcher(settings, other).search("목적", 5)


def test_unknown_ocr_engine_raises_unavailable():
    with pytest.raises(OcrUnavailableError):
        get_ocr_engine("tesseract")


@pytest.mark.slow
def test_bge_m3_real(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        embed_model="BAAI/bge-m3",
        device="auto",
        batch_size=8,
        max_seq_length=1024,
        ocr="none",
    )
    emb = BgeM3Embedder(settings)
    docs = emb.embed_documents(["반려동물을 키울 때 지켜야 할 사육 기준", "주차장 이용 시간과 주차 요금"])
    query = emb.embed_query("반려동물 사육")
    assert len(docs[0]) == 1024
    sim_pet = sum(a * b for a, b in zip(query, docs[0]))
    sim_parking = sum(a * b for a, b in zip(query, docs[1]))
    assert sim_pet > sim_parking


def test_cutoff_drops_vector_candidates(settings, hash_embedder):
    chunks = [
        make_chunk(0, "제1조", "이 규약은 목적을 정한다."),
        make_chunk(1, "제5조", "입주자는 반려동물을 사육할 수 있으며 사육 기준을 따른다."),
    ]
    build(settings, hash_embedder, chunks)
    searcher = Searcher(settings, hash_embedder)
    assert searcher.search("xyz", 5, min_sim=0.5) == []
    assert len(searcher.search("xyz", 5, min_sim=0)) == 2


def test_default_cutoff_comes_from_settings(settings, hash_embedder):
    chunks = [make_chunk(0, "제1조", "이 규약은 목적을 정한다.")]
    build(settings, hash_embedder, chunks)
    strict = dataclasses.replace(settings, min_vec_sim=0.5)
    assert Searcher(strict, hash_embedder).search("xyz", 5) == []


def test_vec_sim_values_and_none_for_bm25_only(settings, hash_embedder):
    chunks = [
        make_chunk(0, "제1조", "이 규약은 목적을 정한다."),
        make_chunk(1, "제5조", "입주자는 반려동물을 사육할 수 있으며 사육 기준을 따른다."),
        make_chunk(2, "제6조", "주차장 이용 시간은 별도로 정한다."),
    ]
    build(settings, hash_embedder, chunks)
    searcher = Searcher(settings, hash_embedder)
    hits = searcher.search(chunks[1].embed_text, 5, min_sim=0)
    assert hits[0].article_ref == "제5조"
    assert hits[0].vec_sim == pytest.approx(1.0, abs=1e-4)
    assert all(-1.0 <= h.vec_sim <= 1.0 for h in hits if h.vec_sim is not None)
    fused = searcher.search(chunks[1].embed_text, 5, min_sim=0.9)
    assert fused[0].article_ref == "제5조"
    assert all(h.vec_sim is None and h.vec_rank is None for h in fused if h.vec_rank is None)


def test_bm25_only_results_dropped_when_no_vector_candidate_passes(settings, hash_embedder):
    chunks = [
        make_chunk(0, "제1조", "이 규약은 목적을 정한다."),
        make_chunk(1, "제5조", "입주자는 반려동물을 사육할 수 있으며 사육 기준을 따른다."),
        make_chunk(2, "제6조", "주차장 이용 시간은 별도로 정한다."),
    ]
    build(settings, hash_embedder, chunks)
    searcher = Searcher(settings, hash_embedder)
    assert searcher.search("반려동물 사육 기준", 5, min_sim=0.999) == []


def test_bm25_only_results_kept_when_cutoff_off(settings, hash_embedder):
    chunks = [
        make_chunk(0, "제1조", "이 규약은 목적을 정한다."),
        make_chunk(1, "제5조", "입주자는 반려동물을 사육할 수 있으며 사육 기준을 따른다."),
        make_chunk(2, "제6조", "주차장 이용 시간은 별도로 정한다."),
    ]
    build(settings, hash_embedder, chunks)
    searcher = Searcher(settings, hash_embedder)
    refs = [h.article_ref for h in searcher.search("반려동물 사육 기준", 5, min_sim=0)]
    assert "제5조" in refs


def test_bm25_search_tokens_matches_search(settings, hash_embedder):
    chunks = [
        make_chunk(0, "제1조", "이 규약은 목적을 정한다."),
        make_chunk(1, "제5조", "입주자는 반려동물을 사육할 수 있으며 사육 기준을 따른다."),
        make_chunk(2, "제6조", "주차장 이용 시간은 별도로 정한다."),
    ]
    build(settings, hash_embedder, chunks)
    _, _, bm, _ = load_index(settings)
    assert bm.search("반려동물 사육", 5) == bm.search_tokens(tokenize("반려동물 사육"), 5)


def test_synonym_questions_get_same_bm25_ranks(settings, hash_embedder):
    chunks = [
        make_chunk(0, "제1조", "이 규약은 목적을 정한다."),
        make_chunk(1, "제93조", "세대 발코니 관리 기준"),
        make_chunk(2, "제19조", "발코니 수선 수선 공사 절차"),
        make_chunk(3, "제6조", "주차장 이용 시간은 별도로 정한다."),
        make_chunk(4, "제7조", "관리비 납부 기한을 정한다."),
    ]
    build(settings, hash_embedder, chunks)
    s = Searcher(settings, hash_embedder, synonyms=Synonyms([["베란다", "발코니"]]))
    q1 = "베란다 수선은 관리사무소에서 해주나요? 아니면 제가 하나요"
    q2 = "발코니 수선은 관리사무소에서 해주나요? 아니면 제가 하나요"
    r1 = {h.chunk_id: h.bm25_rank for h in s.search(q1, 10, min_sim=0)}
    r2 = {h.chunk_id: h.bm25_rank for h in s.search(q2, 10, min_sim=0)}
    assert r1 == r2
    assert r1["d:0001"] is not None


class FixedSimEmbedder:
    model_name = "fake-fixed"

    def __init__(self, sims: dict[str, float], default: float = 0.10):
        self.sims, self.default = sims, default

    def _doc(self, text):
        s = next((v for ref, v in self.sims.items() if f"[{ref}]" in text), self.default)
        return [s, (1 - s * s) ** 0.5, 0.0, 0.0]

    def embed_documents(self, texts):
        return [self._doc(t) for t in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0, 0.0]


def band_setup(settings, sim, query="베란다 수선", **overrides):
    chunks = [
        make_chunk(0, "제1조", "이 규약은 목적을 정한다."),
        make_chunk(1, "제93조", "세대 발코니 관리 기준"),
        make_chunk(2, "제19조", "발코니 수선 수선 공사 절차"),
        make_chunk(3, "제6조", "주차장 이용 시간은 별도로 정한다."),
        make_chunk(4, "제7조", "관리비 납부 기한을 정한다."),
    ]
    emb = FixedSimEmbedder({"제93조": sim, "제19조": 0.40})
    fields = dict(min_vec_sim=0.55, sim_band=0.05, band_top_k=5)
    fields.update(overrides)
    s = dataclasses.replace(settings, **fields)
    build(s, emb, chunks)
    return Searcher(s, emb, synonyms=Synonyms([["베란다", "발코니"]])), query


def test_band_rescues_0_5448_when_bm25_agrees(settings):
    searcher, q = band_setup(settings, 0.5448)
    hits = searcher.search(q, 5)
    assert hits
    assert hits[0].article_ref == "제93조"
    assert hits[0].via_band is True
    assert hits[0].vec_rank == 1
    assert hits[0].vec_sim == pytest.approx(0.5448, abs=1e-3)
    for h in hits[1:]:
        assert h.via_band is False
        assert h.vec_rank is None


def test_strong_pass_0_5514_not_marked_band(settings):
    searcher, q = band_setup(settings, 0.5514)
    hits = searcher.search(q, 5)
    assert hits[0].article_ref == "제93조"
    assert hits[0].via_band is False


def test_band_needs_bm25_evidence(settings):
    searcher, _ = band_setup(settings, 0.5448)
    assert searcher.search("xyz", 5) == []


def test_below_band_stays_empty(settings):
    searcher, q = band_setup(settings, 0.4990)
    assert searcher.search(q, 5) == []


def test_band_disabled_with_zero(settings):
    searcher, q = band_setup(settings, 0.5448, sim_band=0)
    assert searcher.search(q, 5) == []


def test_band_top_k_limits_bm25_rank(settings):
    searcher, q = band_setup(settings, 0.5448, band_top_k=1)
    assert searcher.search(q, 5) == []
    searcher, q = band_setup(settings, 0.5448, band_top_k=2)
    hits = searcher.search(q, 5)
    assert hits[0].via_band is True


def test_synonyms_enable_band_rescue(settings):
    searcher, q = band_setup(settings, 0.5448)
    plain = Searcher(
        dataclasses.replace(settings, min_vec_sim=0.55, sim_band=0.05, band_top_k=5),
        FixedSimEmbedder({"제93조": 0.5448, "제19조": 0.40}),
        synonyms=Synonyms([]),
    )
    assert plain.search(q, 5) == []
    hits = searcher.search(q, 5)
    assert hits
    assert hits[0].via_band is True


def test_min_sim_zero_ignores_band(settings):
    searcher, q = band_setup(settings, 0.5448)
    hits = searcher.search(q, 5, min_sim=0)
    assert all(h.via_band is False for h in hits)
    top = next(h for h in hits if h.article_ref == "제93조")
    assert top.vec_rank == 1


def test_expand_context_adds_referenced_appendix(settings, hash_embedder):
    chunks = [
        make_chunk(0, "제1조", "이 규약은 목적을 정한다."),
        make_chunk(1, "제5조", "입주자는 반려동물을 사육할 수 있으며 [별표 1] 기준을 따른다."),
        dataclasses.replace(make_chunk(2, "별표 1", "관리대상물 목록 표"), part="별표"),
    ]
    build(settings, hash_embedder, chunks)
    searcher = Searcher(settings, hash_embedder)
    hits = searcher.search("반려동물 사육", 1, min_sim=0)
    assert [h.article_ref for h in hits] == ["제5조"]
    expanded = searcher.expand_context(hits)
    appendix = [h for h in expanded if h.article_ref == "별표 1"]
    assert len(appendix) == 1
    assert appendix[0].via_ref is True
    assert appendix[0].text == "관리대상물 목록 표"
    assert expanded[0].via_ref is False
