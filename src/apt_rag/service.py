from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable, Collection, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from apt_rag.generation.answer import DISCLAIMER, NO_HIT_ANSWER, NOT_FOUND_MARK, generate_answer, pure_refusal_text, rewrite_question
from apt_rag.ingest.chunk import build_units, chunk_units, write_chunks_json
from apt_rag.config import Settings
from apt_rag.retrieval.embed import BgeM3Embedder, Embedder
from apt_rag.retrieval.evaluate import evaluate_csv, load_questions, normalize_ref, parse_expected
from apt_rag.ingest.extract import extract_pdf, write_extracted_md, write_pages_json
from apt_rag.generation.llm import LLM, get_llm
from apt_rag.models import AnswerResult, AskEvalResult, AskEvalRow, Chunk, ChatTurn, EvalResult, ExcludedSegment, ExtractResult, GenResult, InspectReport, SearchHit, ValidationReport
from apt_rag.ingest.ocr import OcrUnavailableError, get_ocr_engine
from apt_rag.ingest.pdfinspect import inspect_pdf_file
from apt_rag.generation.qgen import generate, load_chunks
from apt_rag.retrieval.search import Searcher
from apt_rag.retrieval.store import write_index
from apt_rag.ingest.validate import render_report_md, validate


def inspect_pdf(pdf: Path, settings: Settings) -> InspectReport:
    report = inspect_pdf_file(pdf)
    out_dir = settings.work_dir / report.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inspect.json").write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def extract(pdf: Path, settings: Settings, report: InspectReport | None = None) -> ExtractResult:
    pdf = Path(pdf)
    if report is None:
        report = inspect_pdf(pdf, settings)
    ocr = get_ocr_engine(settings.ocr)
    if report.ocr_pages and settings.ocr == "none":
        raise OcrUnavailableError(f"OCR 필요 쪽: {report.ocr_pages}. APT_RAG_OCR 설정 필요")
    result = extract_pdf(pdf, report, ocr)
    out_dir = settings.work_dir / report.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_extracted_md(result, out_dir / "extracted.md")
    write_pages_json(report, result, out_dir / "pages.json")
    return result


@dataclass
class PrepareResult:
    doc_id: str
    work_dir: Path
    page_count: int
    sha256: str
    extract: ExtractResult
    chunks: list[Chunk]
    excluded: list[ExcludedSegment]
    report: ValidationReport


@dataclass
class IndexResult:
    doc_id: str
    work_dir: Path
    chunk_count: int
    report: ValidationReport
    indexed: bool
    aborted_reason: str | None
    dim: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(pdf: Path, settings: Settings) -> PrepareResult:
    pdf = Path(pdf)
    report = inspect_pdf(pdf, settings)
    extract_result = extract(pdf, settings, report)
    units, structure = build_units(extract_result.lines)
    chunks, excluded = chunk_units(units, report.doc_id)
    out_dir = settings.work_dir / report.doc_id
    sha256 = _sha256(pdf)
    write_chunks_json(
        out_dir / "chunks.json",
        report.doc_id,
        str(pdf),
        sha256,
        chunks,
        excluded,
        structure,
    )
    validation = validate(extract_result, report.page_count, units, chunks, excluded, structure)
    (out_dir / "validation_report.json").write_text(
        json.dumps(asdict(validation), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "validation_report.md").write_text(render_report_md(validation), encoding="utf-8")
    return PrepareResult(
        doc_id=report.doc_id,
        work_dir=out_dir,
        page_count=report.page_count,
        sha256=sha256,
        extract=extract_result,
        chunks=chunks,
        excluded=excluded,
        report=validation,
    )


def build_index(
    pdf: Path,
    settings: Settings,
    *,
    no_embed: bool = False,
    strict: bool = False,
    embedder: Embedder | None = None,
) -> IndexResult:
    prep = prepare(pdf, settings)
    if strict and prep.report.has_errors:
        return IndexResult(
            doc_id=prep.doc_id,
            work_dir=prep.work_dir,
            chunk_count=len(prep.chunks),
            report=prep.report,
            indexed=False,
            aborted_reason="검증 오류(--strict)",
        )
    if no_embed:
        return IndexResult(
            doc_id=prep.doc_id,
            work_dir=prep.work_dir,
            chunk_count=len(prep.chunks),
            report=prep.report,
            indexed=False,
            aborted_reason=None,
        )
    embedder = embedder or BgeM3Embedder(settings)
    vectors = embedder.embed_documents([c.embed_text for c in prep.chunks])
    manifest = {
        "doc_id": prep.doc_id,
        "source_pdf": str(pdf),
        "sha256": prep.sha256,
        "embed_model": embedder.model_name,
        "dim": len(vectors[0]) if vectors else 0,
        "chunk_count": len(prep.chunks),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_index(settings, prep.chunks, vectors, manifest)
    return IndexResult(
        doc_id=prep.doc_id,
        work_dir=prep.work_dir,
        chunk_count=len(prep.chunks),
        report=prep.report,
        indexed=True,
        aborted_reason=None,
        dim=manifest["dim"],
    )


def search(query: str, settings: Settings, top_n: int = 5, searcher: Searcher | None = None) -> list[SearchHit]:
    searcher = searcher or Searcher(settings)
    return searcher.search(query, top_n=top_n)


def answer(
    question: str,
    settings: Settings,
    *,
    history: Sequence[ChatTurn] = (),
    top_n: int = 5,
    min_sim: float | None = None,
    searcher: Searcher | None = None,
    llm: LLM | None = None,
) -> AnswerResult:
    question = question.strip()
    if not question:
        raise ValueError("질문이 비어 있습니다")
    searcher = searcher or Searcher(settings)
    used: LLM | None = None
    rewritten: str | None = None
    if history:
        used = llm or get_llm(settings)
        rewritten = rewrite_question(used, history, question)
    query = rewritten or question
    hits = searcher.search(query, top_n=top_n, min_sim=min_sim)
    if not hits:
        return AnswerResult(
            question=question,
            search_query=query,
            rewritten_question=rewritten,
            answer=NO_HIT_ANSWER,
            disclaimer=DISCLAIMER,
            hits=[],
            cited_refs=[],
            unverified_refs=[],
            llm=used.name if used else None,
        )
    context = searcher.expand_context(hits)
    used = used or llm or get_llm(settings)
    gen = generate_answer(used, query, context)
    return AnswerResult(
        question=question,
        search_query=query,
        rewritten_question=rewritten,
        answer=gen.text,
        disclaimer=DISCLAIMER,
        hits=list(context),
        cited_refs=gen.cited_refs,
        unverified_refs=gen.unverified_refs,
        llm=used.name,
        fallback=gen.fallback,
    )


def evaluate(
    csv_path: Path,
    settings: Settings,
    k: int = 5,
    searcher: Searcher | None = None,
    min_sim: float | None = None,
) -> EvalResult:
    searcher = searcher or Searcher(settings)
    result = evaluate_csv(Path(csv_path), searcher, k=k, min_sim=min_sim)
    out_dir = settings.work_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"eval-{stamp}.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


ASK_EVAL_CATEGORIES = (
    "cited",
    "retrieved_not_cited",
    "rank_beyond_k",
    "cut_by_min_sim",
    "not_in_candidates",
    "unanswerable_refused",
    "unanswerable_answered",
)


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def new_ask_eval_partial_path(settings: Settings) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return settings.work_dir / "eval" / f"ask-eval-{safe_model_name(settings.llm_model)}-{stamp}.partial.jsonl"


def evaluate_answers(
    csv_path: Path,
    settings: Settings,
    *,
    k: int = 5,
    min_sim: float | None = None,
    searcher: Searcher | None = None,
    llm: LLM | None = None,
    on_row: Callable[[int, int, AskEvalRow], None] | None = None,
    partial_path: Path | None = None,
) -> AskEvalResult:
    searcher = searcher or Searcher(settings)
    llm = llm or get_llm(settings)
    questions = list(load_questions(Path(csv_path)))
    total = len(questions)
    out_dir = settings.work_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    if partial_path is None:
        partial_path = new_ask_eval_partial_path(settings)
    final_path = partial_path.with_name(partial_path.name.removesuffix(".partial.jsonl") + ".json")
    rows: list[AskEvalRow] = []
    with partial_path.open("w", encoding="utf-8") as partial:
        for i, (question, cell) in enumerate(questions, 1):
            expected = parse_expected(cell)
            res = answer(question, settings, top_n=k, min_sim=min_sim, searcher=searcher, llm=llm)
            refused = NOT_FOUND_MARK in res.answer and not res.cited_refs and pure_refusal_text(res.answer) is not None
            if not expected:
                category = "unanswerable_refused" if refused else "unanswerable_answered"
                gold_rank = None
            else:
                nocut = searcher.search(question, top_n=50, min_sim=0)
                gold_rank = next(
                    (i for i, h in enumerate(nocut, 1) if normalize_ref(h.article_ref) in expected),
                    None,
                )
                hit_n = {normalize_ref(h.article_ref) for h in res.hits}
                cited_n = {normalize_ref(r) for r in res.cited_refs}
                if any(e in cited_n for e in expected):
                    category = "cited"
                elif any(e in hit_n for e in expected):
                    category = "retrieved_not_cited"
                elif gold_rank is None:
                    category = "not_in_candidates"
                elif gold_rank <= k:
                    category = "cut_by_min_sim"
                else:
                    category = "rank_beyond_k"
            row = AskEvalRow(
                question=question,
                expected=expected,
                category=category,
                gold_rank=gold_rank,
                hits=[h.article_ref for h in res.hits],
                cited_refs=list(res.cited_refs),
                unverified_refs=list(res.unverified_refs),
                refused=refused,
                answer=res.answer,
                fallback=res.fallback,
            )
            rows.append(row)
            partial.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            partial.flush()
            if on_row is not None:
                on_row(i, total, row)
    counts = {name: 0 for name in ASK_EVAL_CATEGORIES}
    for row in rows:
        counts[row.category] += 1
    result = AskEvalResult(model=settings.llm_model, k=k, total=len(rows), counts=counts, rows=rows)
    final_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    partial_path.unlink()
    return result


def generate_questions(
    settings: Settings,
    *,
    out: Path,
    count: int = 40,
    per_article: int = 1,
    parts: Collection[str] = ("본문",),
    min_chars: int = 80,
    seed: int = 42,
    model: str | None = None,
    force: bool = False,
    llm: LLM | None = None,
    on_progress: Callable[[int, int, Chunk, int | None], None] | None = None,
) -> GenResult:
    chunks_path = settings.index_dir / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError("색인이 없습니다. 먼저 index 실행: apt-rag index <pdf>")
    chunks = load_chunks(chunks_path)
    if llm is None:
        llm = get_llm(dataclasses.replace(settings, llm_model=model) if model else settings)
    return generate(
        chunks,
        out=Path(out),
        llm=llm,
        count=count,
        per_article=per_article,
        parts=parts,
        min_chars=min_chars,
        seed=seed,
        force=force,
        on_progress=on_progress,
    )
