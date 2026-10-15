import dataclasses

import pytest

from apt_rag.config import Settings
from apt_rag.embed import BgeM3Embedder
from apt_rag.models import Chunk
from apt_rag.search import Searcher, rrf
from apt_rag.store import write_index


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
    hits = Searcher(settings, hash_embedder).search("반려동물 사육", 5)
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
    with pytest.raises(RuntimeError):
        Searcher(settings, other).search("목적", 5)


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
    bm_only = searcher.search("반려동물 사육 기준", 5, min_sim=0.999)
    assert bm_only
    assert all(h.vec_sim is None and h.vec_rank is None for h in bm_only)
