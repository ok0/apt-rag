from __future__ import annotations

import csv
import json
import random
import re
from collections.abc import Callable, Collection, Sequence
from pathlib import Path

from apt_rag.llm import LLM
from apt_rag.models import Chunk, GenResult, GeneratedQuestion

QUOTES = "\"'“”‘’"
MIN_QUESTION_CHARS = 8
MAX_QUESTION_CHARS = 120
MAX_TEXT_CHARS = 1500
TEMPERATURE = 0.7
MIN_MAX_TOKENS = 512
TOKENS_PER_QUESTION = 160
CSV_HEADER = ["question", "expected_article", "source_title", "source_pages"]
_LEADING_MARK = re.compile(r"^\s*(?:\d+\s*[.)]|[-*•])\s*")
_ARTICLE_NO = re.compile(r"제\s*\d+\s*조")

SYSTEM_PROMPT = (
    "당신은 아파트 관리규약 조항을 읽고 입주민 질문을 만드는 도우미입니다.\n"
    "규칙:\n"
    "- 아파트 입주민이 관리사무소에 실제로 물어볼 법한 구어체 질문을 요청된 개수만큼 만드세요.\n"
    "- 질문은 제공된 조항 본문만 읽으면 답할 수 있어야 합니다.\n"
    "- '제N조' 같은 조 번호와 조 제목의 단어를 그대로 쓰지 말고, 일상어로 바꿔 말하세요.\n"
    "- 규약에 없는 내용을 묻거나 추측하지 마세요.\n"
    "- 질문은 한 줄짜리 문장으로 만드세요.\n"
    "- 설명 없이 JSON만 출력하세요. 형식: {\"questions\": [\"질문1\", \"질문2\"]}"
)


def load_chunks(path: Path) -> list[Chunk]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Chunk(**row) for row in rows]


def split_parts(value: str) -> tuple[str, ...]:
    return tuple(piece.strip() for piece in value.split(",") if piece.strip())


def select_chunks(
    chunks: Sequence[Chunk],
    *,
    parts: Collection[str],
    min_chars: int,
    count: int,
    seed: int,
) -> list[Chunk]:
    first_by_ref: dict[str, Chunk] = {}
    for chunk in sorted(chunks, key=lambda c: c.seq):
        if chunk.part not in parts or chunk.char_len < min_chars:
            continue
        first_by_ref.setdefault(chunk.article_ref, chunk)

    rng = random.Random(seed)
    by_chapter: dict[str, list[Chunk]] = {}
    for chunk in first_by_ref.values():
        by_chapter.setdefault(chunk.chapter, []).append(chunk)
    queues = []
    for members in by_chapter.values():
        rng.shuffle(members)
        queues.append(members)

    chosen: list[Chunk] = []
    while queues and len(chosen) < count:
        for queue in queues:
            if queue and len(chosen) < count:
                chosen.append(queue.pop())
        queues = [queue for queue in queues if queue]
    return sorted(chosen, key=lambda c: c.seq)


def build_messages(chunk: Chunk, per_article: int) -> list[dict[str, str]]:
    user = (
        f"질문을 {per_article}개 만드세요.\n\n"
        f"[조항 머리말]\n{chunk.header}\n\n"
        f"[조항 본문]\n{chunk.text[:MAX_TEXT_CHARS]}"
    )
    return [{"role": "user", "content": user}]


def parse_questions(content: str) -> list[str]:
    data = json.loads(content)
    items = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ValueError("questions 배열이 없습니다")
    return [item for item in items if isinstance(item, str)]


def clean_question(raw: str) -> str:
    text = _LEADING_MARK.sub("", raw.strip())
    return text.strip().strip(QUOTES).strip()


def is_valid_question(question: str) -> bool:
    if not MIN_QUESTION_CHARS <= len(question) <= MAX_QUESTION_CHARS:
        return False
    return _ARTICLE_NO.search(question) is None


def _dedupe_key(question: str) -> str:
    return re.sub(r"\s", "", question)


def page_label(chunk: Chunk) -> str:
    if chunk.page_end != chunk.page_start:
        return f"{chunk.page_start}-{chunk.page_end}"
    return str(chunk.page_start)


def write_questions_csv(path: Path, rows: list[GeneratedQuestion]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow([row.question, row.expected_article, row.source_title, row.source_pages])


def generate(
    chunks: Sequence[Chunk],
    *,
    out: Path,
    llm: LLM,
    count: int = 40,
    per_article: int = 1,
    parts: Collection[str] = ("본문",),
    min_chars: int = 80,
    seed: int = 42,
    force: bool = False,
    on_progress: Callable[[int, int, Chunk, int | None], None] | None = None,
) -> GenResult:
    out = Path(out)
    if out.exists() and not force:
        raise FileExistsError(f"이미 존재합니다: {out} (--force로 덮어쓰기)")
    selected = select_chunks(chunks, parts=parts, min_chars=min_chars, count=count, seed=seed)
    llm.check_model()

    questions: list[GeneratedQuestion] = []
    skipped: list[str] = []
    seen: set[str] = set()
    total = len(selected)
    for index, chunk in enumerate(selected, 1):
        content = llm.complete(
            SYSTEM_PROMPT,
            build_messages(chunk, per_article),
            temperature=TEMPERATURE,
            max_tokens=max(MIN_MAX_TOKENS, TOKENS_PER_QUESTION * per_article),
            seed=seed,
            json_mode=True,
        )
        try:
            raw_questions = parse_questions(content)
        except ValueError:
            raw_questions = []
        kept: list[str] = []
        for raw in raw_questions:
            if len(kept) >= per_article:
                break
            cleaned = clean_question(raw)
            if not is_valid_question(cleaned):
                continue
            key = _dedupe_key(cleaned)
            if key in seen:
                continue
            seen.add(key)
            kept.append(cleaned)
        if not kept:
            skipped.append(chunk.article_ref)
            if on_progress:
                on_progress(index, total, chunk, None)
            continue
        for question in kept:
            questions.append(
                GeneratedQuestion(
                    question=question,
                    expected_article=chunk.article_ref,
                    source_title=chunk.title,
                    source_pages=page_label(chunk),
                )
            )
        if on_progress:
            on_progress(index, total, chunk, len(kept))

    write_questions_csv(out, questions)
    return GenResult(out_path=out, selected=total, questions=questions, skipped=skipped)
