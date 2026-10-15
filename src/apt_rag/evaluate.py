from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

from apt_rag.models import EvalFailure, EvalResult


def normalize_ref(s: str) -> str:
    s = re.sub(r"\s", "", unicodedata.normalize("NFC", s))
    return re.sub(r"(조(?:의\d+)?)제\d+항.*$", r"\1", s)


def _split_cell(cell: str) -> list[str]:
    return [piece.strip() for piece in re.split(r"[;|]", cell) if piece.strip()]


def parse_expected(cell: str) -> list[str]:
    return [normalize_ref(piece) for piece in _split_cell(cell)]


def load_questions(csv_path: Path) -> list[tuple[str, list[str]]]:
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if "question" not in fields or "expected_article" not in fields:
            raise ValueError("CSV에 question, expected_article 컬럼이 필요합니다")
        rows = []
        for row in reader:
            question = (row.get("question") or "").strip()
            if not question:
                continue
            rows.append((question, row.get("expected_article") or ""))
    return [(q, cell) for q, cell in rows]


def evaluate_csv(csv_path: Path, searcher, k: int = 5) -> EvalResult:
    questions = load_questions(Path(csv_path))
    total = len(questions)
    hits = 0
    failures: list[EvalFailure] = []
    for question, cell in questions:
        results = searcher.search(question, top_n=k)
        got = [normalize_ref(h.article_ref) for h in results]
        expected = parse_expected(cell)
        if any(e in got for e in expected):
            hits += 1
        else:
            failures.append(
                EvalFailure(
                    question=question,
                    expected=_split_cell(cell),
                    got=[h.article_ref for h in results],
                )
            )
    hit_rate = hits / total if total else 0.0
    return EvalResult(k=k, total=total, hits=hits, hit_rate=hit_rate, failures=failures)
