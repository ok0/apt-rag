from __future__ import annotations

import argparse

from apt_rag import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apt-rag")
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    print("미구현")
    return 2
