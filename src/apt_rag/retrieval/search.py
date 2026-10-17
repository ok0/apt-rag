from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING

from apt_rag.retrieval.bm25 import BM25Index, tokenize
from apt_rag.config import Settings
from apt_rag.retrieval.embed import BgeM3Embedder, Embedder
from apt_rag.models import Chunk, SearchHit
from apt_rag.retrieval.synonyms import Synonyms, load_synonyms
from apt_rag.retrieval.store import IndexMismatchError, load_index

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

RRF_K = 60
CANDIDATES = 50
CONTEXT_ARTICLE_CHARS = 2000
CONTEXT_TOTAL_CHARS = 12000
CONTEXT_MAX_REFS = 2
_REF_APPX = re.compile(r"별표\s*(\d+)")
_REF_ART = re.compile(r"제(\d+)조(?:의(\d+))?")
_LAW_CONTEXT_CHARS = "법령칙」"


def _is_law_ref(text: str, start: int) -> bool:
    j = start - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    return j >= 0 and text[j] in _LAW_CONTEXT_CHARS


def _context_text(group: list[Chunk], chunk_id: str) -> str:
    full = "\n".join(c.text for c in group)
    if len(full) <= CONTEXT_ARTICLE_CHARS:
        return full
    text = group[0].text
    if chunk_id != group[0].chunk_id:
        matched = next(c for c in group if c.chunk_id == chunk_id)
        text += "\n…\n" + matched.text
    return text


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
        self._loaded: tuple[Collection, dict[str, Chunk], BM25Index] | None = None
        self._by_ref: dict[str, list[Chunk]] | None = None

    def _ensure(self) -> tuple[Collection, dict[str, Chunk], BM25Index]:
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

    def _expanded_sims(self, collection: Collection, query: str, tokens: list[str], n: int) -> dict[str, float]:
        terms = self._synonyms.expansion_terms(tokens)
        if not terms:
            return {}
        res = collection.query(
            query_embeddings=[self._embedder.embed_query(query + " " + " ".join(terms))],
            n_results=n,
            include=["distances"],
        )
        return {cid: 1.0 - float(dist) for cid, dist in zip(res["ids"][0], res["distances"][0])}

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
        q_tokens = tokenize(query)
        boosted = {
            cid: s
            for cid, s in self._expanded_sims(collection, query, q_tokens, n).items()
            if s >= cutoff and s > sims.get(cid, -1.0)
        }
        if boosted:
            sims.update(boosted)
            ids = sorted(sims, key=lambda c: sims[c], reverse=True)[:n]
        bm_tokens = self._synonyms.expand_tokens(q_tokens)
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

    def _chunks_by_ref(self) -> dict[str, list[Chunk]]:
        if self._by_ref is None:
            _, chunks, _ = self._ensure()
            by_ref: dict[str, list[Chunk]] = {}
            for chunk in chunks.values():
                by_ref.setdefault(chunk.article_ref, []).append(chunk)
            for group in by_ref.values():
                group.sort(key=lambda c: c.seq)
            self._by_ref = by_ref
        return self._by_ref

    def expand_context(self, hits: list[SearchHit]) -> list[SearchHit]:
        by_ref = self._chunks_by_ref()
        result: list[SearchHit] = []
        for hit in hits:
            group = by_ref[hit.article_ref]
            result.append(dataclasses.replace(hit, text=_context_text(group, hit.chunk_id)))
        total = sum(len(h.text) for h in result)
        if total > CONTEXT_TOTAL_CHARS:
            return result
        present = {h.article_ref for h in hits}
        appx: list[str] = []
        art: list[str] = []
        for hit in result:
            for m in _REF_APPX.finditer(hit.text):
                if not _is_law_ref(hit.text, m.start()):
                    appx.append(f"별표 {m.group(1)}")
            if hit.part in ("본문", "별표"):
                for m in _REF_ART.finditer(hit.text):
                    if not _is_law_ref(hit.text, m.start()):
                        ref = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
                        art.append(ref)
        candidates: list[str] = []
        for ref in appx + art:
            if ref in by_ref and ref not in present and ref not in candidates:
                candidates.append(ref)
        for ref in candidates[:CONTEXT_MAX_REFS]:
            group = by_ref[ref]
            body = _context_text(group, group[0].chunk_id)
            if total + len(body) > CONTEXT_TOTAL_CHARS:
                break
            total += len(body)
            first = group[0]
            result.append(
                SearchHit(
                    rank=len(result) + 1,
                    article_ref=first.article_ref,
                    part=first.part,
                    title=first.title,
                    header=first.header,
                    chapter=first.chapter,
                    page_start=first.page_start,
                    page_end=first.page_end,
                    score=0.0,
                    vec_rank=None,
                    bm25_rank=None,
                    vec_sim=None,
                    chunk_id=first.chunk_id,
                    text=body,
                    via_ref=True,
                )
            )
        return result
