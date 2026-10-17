from pathlib import Path

import pytest

from apt_rag.retrieval.bm25 import tokenize
from apt_rag.retrieval.synonyms import Synonyms, load_synonyms, parse_synonyms


def test_parse_skips_comments_and_blank_lines():
    text = "# 주석\n\n베란다 = 발코니  # 끝주석\n엘베=승강기=엘리베이터\n"
    assert parse_synonyms(text) == [["베란다", "발코니"], ["엘베", "승강기", "엘리베이터"]]


def test_parse_dedupes_and_drops_empty_terms():
    assert parse_synonyms("a==b=a") == [["a", "b"]]


def test_parse_single_term_raises_with_line_number():
    with pytest.raises(ValueError) as exc:
        parse_synonyms("베란다=발코니\n베란다\n", source="s.txt")
    assert "s.txt:2" in str(exc.value)


def test_expand_adds_other_members():
    assert Synonyms([["베란다", "발코니"]]).expand_tokens(["베란다", "수선"]) == ["베란다", "수선", "발코니"]


def test_expand_no_duplicates_when_both_present():
    assert Synonyms([["베란다", "발코니"]]).expand_tokens(["베란다", "발코니"]) == ["베란다", "발코니"]


def test_expand_multi_token_member():
    syn = Synonyms([["반상회", "입주자대표회의"]])
    assert syn.expand_tokens(["반상회"]) == ["반상회", "입주자", "대표", "회의"]
    assert syn.expand_tokens(["입주자", "대표", "회의"]) == ["입주자", "대표", "회의", "반상회"]
    assert syn.expand_tokens(["대표", "입주자"]) == ["대표", "입주자"]


def test_expand_empty_synonyms_identity():
    assert Synonyms([]).expand_tokens(["a"]) == ["a"]


def test_balcony_veranda_questions_expand_to_same_tokens():
    q1 = "베란다 수선은 관리사무소에서 해주나요? 아니면 제가 하나요"
    q2 = q1.replace("베란다", "발코니")
    syn = Synonyms([["베란다", "발코니"]])
    assert sorted(syn.expand_tokens(tokenize(q1))) == sorted(syn.expand_tokens(tokenize(q2)))


def test_load_missing_file_is_empty(tmp_path):
    assert len(load_synonyms(tmp_path / "none.txt")) == 0
    assert len(load_synonyms(None)) == 0


def test_repo_seed_file_is_valid():
    syn = load_synonyms(Path(__file__).parents[1] / "synonyms.txt")
    assert 20 <= len(syn) <= 30
    assert any({"베란다", "발코니"} <= set(g) for g in syn.groups)


def test_repo_seed_maps_pet_words_to_gachuk():
    syn = load_synonyms(Path(__file__).parents[1] / "synonyms.txt")
    for q in ("반려동물 키워도 되나요", "강아지 키워도 되나요", "반려견 키워도 되나요", "고양이 키워도 되나요"):
        assert "가축" in syn.expand_tokens(tokenize(q))


def test_expansion_terms_returns_missing_members():
    assert Synonyms([["베란다", "발코니"]]).expansion_terms(["베란다", "수선"]) == ["발코니"]


def test_expansion_terms_empty_when_all_present_or_no_match():
    syn = Synonyms([["베란다", "발코니"]])
    assert syn.expansion_terms(["베란다", "발코니"]) == []
    assert syn.expansion_terms(["주차장"]) == []


def test_expansion_terms_multi_token_member():
    syn = Synonyms([["반상회", "입주자대표회의"]])
    assert syn.expansion_terms(["반상회"]) == ["입주자대표회의"]
    assert syn.expansion_terms(["입주자", "대표", "회의"]) == ["반상회"]
    assert syn.expansion_terms(["대표", "입주자"]) == []


def test_expansion_terms_empty_synonyms():
    assert Synonyms([]).expansion_terms(["a"]) == []
