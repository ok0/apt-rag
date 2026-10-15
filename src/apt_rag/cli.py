from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from apt_rag import __version__
from apt_rag.config import load_settings
from apt_rag.ocr import OcrUnavailableError
from apt_rag.qgen import OllamaError, split_parts
from apt_rag.search import Searcher
from apt_rag.service import build_index, evaluate, generate_questions, inspect_pdf


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apt-rag",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  apt-rag inspect data/bylaws.pdf\n"
            "  apt-rag index data/bylaws.pdf\n"
            '  apt-rag search "반려동물 키워도 되나요"\n'
            "  apt-rag eval eval/sample_questions.csv\n"
            "  apt-rag help search"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="PDF 쪽별 텍스트/이미지 판별")
    p_inspect.add_argument("pdf")
    p_inspect.add_argument("--json", action="store_true")

    p_index = sub.add_parser("index", help="PDF를 추출/청크/검증/색인")
    p_index.add_argument("pdf")
    p_index.add_argument("--no-embed", action="store_true")
    p_index.add_argument("--strict", action="store_true")

    p_search = sub.add_parser("search", help="색인에서 검색")
    p_search.add_argument("query")
    p_search.add_argument("--top-n", type=int, default=5)
    p_search.add_argument("--excerpt", type=int, default=160)
    p_search.add_argument("--json", action="store_true")

    p_eval = sub.add_parser("eval", help="질문 CSV로 검색 적중률 평가")
    p_eval.add_argument("csv")
    p_eval.add_argument("-k", type=int, default=5)
    p_eval.add_argument("--json", action="store_true")

    p_gen = sub.add_parser("gen-questions", help="색인 청크로 평가 질문 CSV 초안 생성 (로컬 ollama)")
    p_gen.add_argument("--count", type=int, default=40, help="질문을 만들 조항 수 (기본 40)")
    p_gen.add_argument("--per-article", type=int, default=1, help="조항당 질문 수 (기본 1)")
    p_gen.add_argument(
        "--out", default="eval/generated_questions.csv", help="출력 CSV 경로 (기본 eval/generated_questions.csv)"
    )
    p_gen.add_argument("--model", default=None, help="ollama 모델 이름 (기본 APT_RAG_LLM_MODEL 또는 qwen2.5:7b)")
    p_gen.add_argument("--seed", type=int, default=42, help="조항 샘플링과 생성 시드 (기본 42)")
    p_gen.add_argument("--parts", default="본문", help="대상 부분, 쉼표로 여러 개 (예: 본문,별표; 기본 본문)")
    p_gen.add_argument("--min-chars", type=int, default=80, help="청크 최소 글자 수 (기본 80)")
    p_gen.add_argument("--force", action="store_true", help="기존 출력 파일을 덮어쓰기")

    p_help = sub.add_parser("help", help="도움말 표시 (help <명령>으로 명령별 도움말)")
    p_help.add_argument("topic", nargs="?", help="도움말을 볼 명령 이름")

    return parser


def _cmd_inspect(args: argparse.Namespace) -> int:
    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"파일을 찾을 수 없습니다: {pdf}", file=sys.stderr)
        return 1
    settings = load_settings()
    report = inspect_pdf(pdf, settings)
    if args.json:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
        return 0
    print("쪽  판별   글자수  이미지비율  표  비고")
    for page in report.pages:
        print(
            f"{page.page:<4} {page.kind:<6} {page.text_chars:>6}  "
            f"{page.image_coverage:>8.2f}  {page.table_count:>2}  {page.note}"
        )
    counts = " / ".join(f"{kind}={count}" for kind, count in report.counts.items())
    print(f"판별 개수: {counts}")
    print(report.recommendation)
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    pdf = Path(args.pdf)
    if not pdf.exists():
        print(f"파일을 찾을 수 없습니다: {pdf}", file=sys.stderr)
        return 1
    settings = load_settings()
    try:
        result = build_index(pdf, settings, no_embed=args.no_embed, strict=args.strict)
    except OcrUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 4
    report = result.report
    print("[1/5] 판별 완료")
    print("[2/5] 추출 완료")
    print(f"[3/5] 청킹/검증: 청크 {result.chunk_count}개, 제외 {sum(report.summary.get('excluded_by_reason', {}).values())}개")
    print("      요약: " + json.dumps(report.summary, ensure_ascii=False))
    shown = {"ERROR": 0, "WARN": 0}
    for issue in report.issues:
        if issue.severity not in shown:
            continue
        if shown[issue.severity] >= 10:
            continue
        shown[issue.severity] += 1
        print(f"      [{issue.severity}] {issue.code}: {issue.message}")
    for severity in ("ERROR", "WARN"):
        total = sum(1 for i in report.issues if i.severity == severity)
        if total > 10:
            print(f"      [{severity}] 외 {total - 10}건")
    if args.no_embed:
        print("[4/5] 임베딩: 생략 (--no-embed)")
    else:
        print(f"[4/5] 임베딩 완료 (차원 {result.dim})")
    print("[5/5] 저장 완료" if result.indexed else "[5/5] 저장: 건너뜀")
    print(f"산출물: {result.work_dir}")
    if result.aborted_reason:
        print(f"중단: {result.aborted_reason}", file=sys.stderr)
        return 3
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    settings = load_settings()
    try:
        hits = Searcher(settings).search(args.query, top_n=args.top_n)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps([asdict(h) for h in hits], ensure_ascii=False, indent=2))
        return 0
    for hit in hits:
        pages = f"p.{hit.page_start}" + (f"-{hit.page_end}" if hit.page_end != hit.page_start else "")
        meta = [hit.part] + ([f"{hit.chapter}"] if hit.chapter else []) + [pages]
        print(
            f"{hit.rank}. {hit.article_ref} {hit.title}  [{' · '.join(meta)}]  "
            f"score={hit.score:.4f} (vec #{hit.vec_rank or '-'}, bm25 #{hit.bm25_rank or '-'})"
        )
        flat = " ".join(hit.text.split())
        print(f"   {flat[: args.excerpt]}…")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"파일을 찾을 수 없습니다: {csv_path}", file=sys.stderr)
        return 1
    settings = load_settings()
    try:
        result = evaluate(csv_path, settings, k=args.k)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
        return 0
    print(f"hit@{result.k}: {result.hits}/{result.total} = {result.hit_rate:.1%}")
    if result.failures:
        print(f"실패 {len(result.failures)}건:")
        for failure in result.failures:
            print(
                f"- Q: {failure.question} / 정답: {', '.join(failure.expected)} / "
                f"상위{result.k}: {', '.join(failure.got)}"
            )
    print(f"저장: {settings.work_dir / 'eval'}")
    return 0


def _cmd_gen_questions(args: argparse.Namespace) -> int:
    settings = load_settings()

    def progress(index: int, total: int, chunk, produced: int | None) -> None:
        label = f"{chunk.article_ref} {chunk.title}"
        if produced is None:
            print(f"[{index}/{total}] {label} -> 건너뜀 (유효한 질문 없음)")
        else:
            print(f"[{index}/{total}] {label} -> {produced}개 질문")

    try:
        result = generate_questions(
            settings,
            out=Path(args.out),
            count=args.count,
            per_article=args.per_article,
            parts=split_parts(args.parts),
            min_chars=args.min_chars,
            seed=args.seed,
            model=args.model,
            force=args.force,
            on_progress=progress,
        )
    except (FileNotFoundError, FileExistsError, OllamaError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"생성 {len(result.questions)}개, 건너뜀 {len(result.skipped)}개, 저장: {result.out_path}")
    print("정답 조항은 자동 부여된 초안이므로 사람이 검토하세요.")
    return 0


def _cmd_help(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if not args.topic:
        parser.print_help()
        return 0
    choices = next(
        a.choices for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    if args.topic not in choices:
        print(f"알 수 없는 명령: {args.topic} (가능: {', '.join(choices)})", file=sys.stderr)
        return 1
    choices[args.topic].print_help()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect":
        return _cmd_inspect(args)
    if args.command == "index":
        return _cmd_index(args)
    if args.command == "search":
        return _cmd_search(args)
    if args.command == "eval":
        return _cmd_eval(args)
    if args.command == "gen-questions":
        return _cmd_gen_questions(args)
    if args.command == "help":
        return _cmd_help(parser, args)
    print("미구현")
    return 2
