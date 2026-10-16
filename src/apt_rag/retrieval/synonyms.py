from __future__ import annotations

import unicodedata
from pathlib import Path

from apt_rag.retrieval.bm25 import tokenize


def parse_synonyms(text: str, source: str = "<synonyms>") -> list[list[str]]:
    groups: list[list[str]] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = unicodedata.normalize("NFC", raw.split("#", 1)[0]).strip()
        if not line:
            continue
        terms: list[str] = []
        for piece in line.split("="):
            term = " ".join(piece.split())
            if term and term not in terms:
                terms.append(term)
        if len(terms) < 2:
            raise ValueError(
                f"{source}:{lineno}: 동의어 묶음에는 '='로 구분된 단어가 2개 이상 필요합니다: {raw.strip()}"
            )
        groups.append(terms)
    return groups


def _contains(seq: list[str], sub: tuple[str, ...]) -> bool:
    if not sub:
        return False
    n = len(sub)
    return any(tuple(seq[i : i + n]) == sub for i in range(len(seq) - n + 1))


class Synonyms:
    def __init__(self, groups: list[list[str]]) -> None:
        self.groups = [list(g) for g in groups]
        self._token_groups = [
            [t for t in (tuple(tokenize(term)) for term in g) if t] for g in self.groups
        ]

    def __len__(self) -> int:
        return len(self.groups)

    def expand_tokens(self, tokens: list[str]) -> list[str]:
        result = list(tokens)
        have = set(tokens)
        for token_group in self._token_groups:
            if not any(_contains(tokens, member) for member in token_group):
                continue
            for member in token_group:
                for tok in member:
                    if tok not in have:
                        have.add(tok)
                        result.append(tok)
        return result


def load_synonyms(path: Path | None) -> Synonyms:
    if path is None or not Path(path).exists():
        return Synonyms([])
    return Synonyms(parse_synonyms(Path(path).read_text(encoding="utf-8"), source=str(path)))
