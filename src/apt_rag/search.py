from __future__ import annotations

from apt_rag.bm25 import BM25Index, tokenize
from apt_rag.config import Settings
from apt_rag.embed import BgeM3Embedder, Embedder
from apt_rag.models import Chunk, SearchHit
from apt_rag.synonyms import Synonyms, load_synonyms
from apt_rag.store import IndexMismatchError, load_index

RRF_K = 60
CANDIDATES = 50


def rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank)
    return score


class Searcher:
    def __init__(
        self, settings: Settings, embedder: Embedder | None = None, synonyms: Synonyms | None = None
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._synonyms = synonyms
        self._loaded = None

    def _ensure(self):
        if self._loaded is None:
            collection, chunks, bm25, manifest = load_index(self._settings)
            if self._synonyms is None:
                self._synonyms = load_synonyms(self._settings.synonyms_path)
            if self._embedder is None:
                self._embedder = BgeM3Embedder(self._settings)
            if self._embedder.model_name != manifest["embed_model"]:
                raise IndexMismatchError(
                    f"임베딩 모델 불일치: 색인={manifest['embed_model']}, 현재={self._embedder.model_name}. index를 다시 실행하거나 APT_RAG_EMBED_MODEL을 맞추세요"
                )
            self._loaded = (collection, chunks, bm25)
        return self._loaded

    def _band_candidates(self, ids: list[str], sims: dict[str, float], bm_ids: list[str], cutoff: float) -> list[str]:
        band = self._settings.sim_band
        k = self._settings.band_top_k
        if band <= 0 or k <= 0:
            return []
        lower = round(cutoff - band, 6)
        bm_top = set(bm_ids[:k])
        return [cid for cid in ids[:k] if sims[cid] >= lower and cid in bm_top]

    def search(self, query: str, top_n: int = 5, min_sim: float | None = None) -> list[SearchHit]:
        collection, chunks, bm25 = self._ensure()
        cutoff = self._settings.min_vec_sim if min_sim is None else min_sim
        n = min(CANDIDATES, collection.count())
        if n == 0:
            return []
        res = collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=n,
            include=["distances"],
        )
        ids = list(res["ids"][0])
        sims = {cid: 1.0 - float(dist) for cid, dist in zip(ids, res["distances"][0])}
        bm_tokens = self._synonyms.expand_tokens(tokenize(query))
        bm_ids = [cid for cid, _ in bm25.search_tokens(bm_tokens, CANDIDATES)]
        band_ids: set[str] = set()
        if cutoff > 0:
            vec_ids = [cid for cid in ids if sims[cid] >= cutoff]
            if not vec_ids:
                vec_ids = self._band_candidates(ids, sims, bm_ids, cutoff)
                band_ids = set(vec_ids)
            if not vec_ids:
                return []
        else:
            vec_ids = ids
        scores = rrf([vec_ids, bm_ids])
        vec_pos = {cid: i for i, cid in enumerate(ids, start=1)}
        vec_rank = {cid: vec_pos[cid] for cid in vec_ids}
        vec_sim = {cid: sims[cid] for cid in vec_ids}
        bm_rank = {cid: i for i, cid in enumerate(bm_ids, start=1)}
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        seen: set[str] = set()
        hits: list[SearchHit] = []
        for cid, score in ordered:
            chunk: Chunk = chunks[cid]
            if chunk.article_ref in seen:
                continue
            seen.add(chunk.article_ref)
            hits.append(
                SearchHit(
                    rank=len(hits) + 1,
                    article_ref=chunk.article_ref,
                    part=chunk.part,
                    title=chunk.title,
                    header=chunk.header,
                    chapter=chunk.chapter,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    score=score,
                    vec_rank=vec_rank.get(cid),
                    bm25_rank=bm_rank.get(cid),
                    vec_sim=vec_sim.get(cid),
                    chunk_id=cid,
                    text=chunk.text,
                    via_band=cid in band_ids,
                )
            )
            if len(hits) >= top_n:
                break
        return hits
