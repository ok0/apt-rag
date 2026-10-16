from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def count_content_chars(s: str) -> int:
    return len(re.findall(r"[가-힣A-Za-z0-9]", s))


def page_range(start: int, end: int, prefix: str = "p.") -> str:
    label = f"{prefix}{start}"
    if end != start:
        label += f"-{end}"
    return label


@dataclass
class PageInfo:
    page: int
    kind: str
    text_chars: int
    image_coverage: float
    table_count: int
    garbled: bool
    note: str


@dataclass
class InspectReport:
    pdf: str
    doc_id: str
    page_count: int
    pages: list[PageInfo]
    counts: dict[str, int]
    ocr_pages: list[int]
    recommendation: str


@dataclass
class Line:
    text: str
    page: int
    page_end: int
    kind: str = "text"
    para_break: bool = False
    lead_space: bool = False
    trail_space: bool = False


@dataclass
class RemovedLine:
    page: int
    text: str
    reason: str


@dataclass
class ExtractResult:
    doc_id: str
    lines: list[Line]
    removed: list[RemovedLine]
    raw_chars: int
    extracted_chars: int
    removed_chars: int
    table_count: int


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    seq: int
    part: str
    scope: str
    scope_label: str
    chapter: str
    section: str
    article_ref: str
    article_no: str
    article_title: str
    paragraph_no: int | None
    split_index: int | None
    split_total: int | None
    appendix_label: str
    title: str
    page_start: int
    page_end: int
    header: str
    text: str
    embed_text: str
    char_len: int


@dataclass
class ExcludedSegment:
    reason: str
    scope: str
    article_ref: str
    page_start: int
    page_end: int
    char_len: int
    text: str


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    detail: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    summary: dict
    scopes: dict
    issues: list[Issue]
    has_errors: bool


@dataclass
class SearchHit:
    rank: int
    article_ref: str
    part: str
    title: str
    header: str
    chapter: str
    page_start: int
    page_end: int
    score: float
    vec_rank: int | None
    bm25_rank: int | None
    vec_sim: float | None
    chunk_id: str
    text: str
    via_band: bool = False


@dataclass
class EvalFailure:
    question: str
    expected: list[str]
    got: list[str]


@dataclass
class EvalResult:
    k: int
    total: int
    hits: int
    hit_rate: float
    failures: list[EvalFailure]


@dataclass
class AskEvalRow:
    question: str
    expected: list[str]
    category: str
    gold_rank: int | None
    hits: list[str]
    cited_refs: list[str]
    unverified_refs: list[str]
    refused: bool
    answer: str


@dataclass
class AskEvalResult:
    k: int
    total: int
    counts: dict[str, int]
    rows: list[AskEvalRow]


@dataclass
class GeneratedQuestion:
    question: str
    expected_article: str
    source_title: str
    source_pages: str


@dataclass
class GenResult:
    out_path: Path
    selected: int
    questions: list[GeneratedQuestion]
    skipped: list[str]


@dataclass
class ChatTurn:
    role: str      # "user" | "assistant"
    content: str


@dataclass
class AnswerResult:
    question: str                     # 입력 원문(strip)
    search_query: str                 # 실제 검색/답변에 쓴 질문
    rewritten_question: str | None    # 이력이 없으면 None
    answer: str
    disclaimer: str
    hits: list[SearchHit]
    cited_refs: list[str]             # 답변이 인용했고 검색 결과에 있는 article_ref (hits 순서)
    unverified_refs: list[str]        # 답변이 인용했지만 검색 결과에 없는 표기(공백 제거형)
    llm: str | None                   # 호출한 LLM 이름, 호출 안 했으면 None
