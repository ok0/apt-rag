from apt_rag.chunk import build_units, chunk_units
from apt_rag.extract import is_appendix_heading
from apt_rag.models import Line


def L(text, page=1):
    return Line(text=text, page=page, page_end=page)


def run(lines):
    units, structure = build_units(lines)
    chunks, excluded = chunk_units(units, "d")
    return units, chunks, excluded, structure


def test_toc_and_preamble_excluded_and_chapter_from_heading():
    lines = [
        L("제1조(목적) ········ 1"),
        L("제2조(정의) ········ 2"),
        L("표지 텍스트"),
        L("제 1 장  총   칙"),
        L("제1조(목적)"),
        L("이 규약은 목적을 정한다."),
        L("제2조(정의)"),
        L("정의 본문."),
    ]
    _, chunks, excluded, structure = run(lines)
    assert [e.reason for e in excluded] == ["preamble"]
    first = next(c for c in chunks if c.article_ref == "제1조")
    assert first.chapter == "제1장 총칙"
    assert first.header == "[제1장 총칙 > 제1조(목적)]"


def test_article_like_line_inside_text_is_not_new_article():
    lines = [
        L("제1조(목적)"),
        L("본문 A."),
        L("제23조(입주자등의 의무)에 따라 처리한다."),
    ]
    _, chunks, _, _ = run(lines)
    assert [c.article_ref for c in chunks] == ["제1조"]


def test_deleted_articles_are_excluded():
    lines = [
        L("제1조(목적)"),
        L("본문."),
        L("제3조 삭제 <2020. 1. 1.>"),
        L("제4조(삭제)"),
        L("제5조(사용) 사용 기준을 정한다."),
    ]
    _, chunks, excluded, _ = run(lines)
    assert [e.reason for e in excluded] == ["deleted", "deleted"]
    assert [e.article_ref for e in excluded] == ["제3조", "제4조"]
    assert "제3조" not in [c.article_ref for c in chunks]


def test_article_with_branch_number():
    lines = [
        L("제1조(목적)"),
        L("본문."),
        L("제4조의2(테스트) 내용입니다."),
    ]
    _, chunks, _, _ = run(lines)
    chunk = next(c for c in chunks if c.article_ref == "제4조의2")
    assert chunk.article_no == "4의2"


def test_long_article_split_by_paragraph():
    lines = [
        L("제1조(목적)"),
        L("본문."),
        L("제 2 장  공동생활의 질서"),
        L("제5조(반려동물 사육) ① " + "가" * 600),
        L("② " + "나" * 600),
        L("③ " + "다" * 600),
    ]
    _, chunks, _, _ = run(lines)
    art = [c for c in chunks if c.article_ref == "제5조"]
    assert [c.paragraph_no for c in art] == [1, 2, 3]
    assert art[0].text.startswith("제5조(반려동물 사육) ①")
    assert art[1].header.startswith("[제2장 공동생활의 질서 > 제5조(반려동물 사육) > 제2항")


def test_long_article_without_paragraphs_split_evenly():
    lines = [
        L("제1조(목적)"),
        L("본문."),
        L("제6조(시설) 관리 기준"),
        L("가" * 500),
        L("나" * 500),
        L("다" * 500),
        L("라" * 500),
    ]
    _, chunks, _, _ = run(lines)
    art = [c for c in chunks if c.article_ref == "제6조"]
    assert len(art) == 2
    assert art[0].split_total == 2
    assert art[0].header.endswith("(1/2)]")


def test_long_article_without_paragraphs_split_by_item():
    lines = [
        L("제1조(목적)"),
        L("본문."),
        L("제83조(동의기준) 동의기준은 다음 각 호와 같다. 1. " + "가" * 600),
        L("2. " + "나" * 600),
        L("1) 세부 사항"),
        L("2.5톤 이상 화물차 " + "라" * 50),
        L("3. 가축의 사육 " + "다" * 400),
    ]
    _, chunks, _, _ = run(lines)
    art = [c for c in chunks if c.article_ref == "제83조"]
    assert len(art) == 3
    assert art[0].text.startswith("제83조(동의기준)")
    assert art[0].header == "[제83조(동의기준) > 제1호 (1/3)]"
    assert art[1].text.startswith("2. ")
    assert "1) 세부 사항" in art[1].text and "2.5톤" in art[1].text
    assert art[2].text.startswith("3. 가축의 사육")
    assert art[2].header == "[제83조(동의기준) > 제3호 (3/3)]"
    assert all(c.paragraph_no is None for c in art)


def test_item_split_merges_article_head_into_first_item():
    lines = [
        L("제1조(목적)"),
        L("본문."),
        L("제3조(정의) 용어의 정의는 다음과 같다."),
        L("1. " + "가" * 800),
        L("2. " + "나" * 800),
    ]
    _, chunks, _, _ = run(lines)
    art = [c for c in chunks if c.article_ref == "제3조"]
    assert len(art) == 2
    assert art[0].text.startswith("제3조(정의)") and "1. 가" in art[0].text
    assert art[0].header == "[제3조(정의) > 제1호 (1/2)]"
    assert art[1].header == "[제3조(정의) > 제2호 (2/2)]"


def test_addenda_scopes_and_refs():
    lines = [
        L("제1조(목적)"),
        L("본문."),
        L("부  칙"),
        L("제1조(시행일) 이 규약은 공포한 날부터 시행한다."),
        L("부칙 <2021. 5. 1.>"),
        L("제1조(경과조치) 종전 규정에 따른다."),
    ]
    _, chunks, _, _ = run(lines)
    scopes = {c.scope for c in chunks}
    assert {"부칙#1", "부칙#2"} <= scopes
    refs = [c.article_ref for c in chunks if c.scope.startswith("부칙")]
    assert refs == ["부칙 제1조", "부칙 제1조"]


def test_appendix_heading_creates_appendix_unit():
    lines = [
        L("제1조(목적)"),
        L("본문."),
        L("[별표 1] 관리대상물(제4조 관련)"),
        L("제1조(목적) 별표 안의 줄"),
        L("표 내용"),
    ]
    _, chunks, _, _ = run(lines)
    appendix = next(c for c in chunks if c.part == "별표")
    assert appendix.article_ref == "별표 1"
    assert appendix.title == "관리대상물(제4조 관련)"
    assert len([c for c in chunks if c.article_no == "1"]) == 1


def test_byeolcheom_with_article_header():
    lines = [
        L("제1조(목적)"),
        L("본문."),
        L("[별첨 2](제29조제6항 관련)"),
        L("입주자대표회의 감사 업무규정"),
        L("제 1 장  총 칙"),
        L("제1조 (목적) 이 규정은 감사를 정한다."),
    ]
    _, chunks, _, _ = run(lines)
    art = next(c for c in chunks if c.article_ref == "별첨 2 제1조")
    assert art.part == "별첨"
    assert art.header.startswith(
        "[별첨 2 입주자대표회의 감사 업무규정 (제29조제6항 관련) > 제1장 총칙 > 제1조(목적)]"
    )


def test_appendix_form_has_no_articles():
    lines = [
        L("제1조(목적)"),
        L("본문."),
        L("[별첨2 제1호서식]"),
        L("제2조(원래) 서식 안의 조문 같은 줄"),
    ]
    _, chunks, _, _ = run(lines)
    form = next(c for c in chunks if c.appendix_label == "별첨2 제1호서식")
    assert form.part == "별첨"
    assert len([c for c in chunks if c.article_no == "2"]) == 0


def test_bracket_sentence_is_not_appendix_heading():
    assert is_appendix_heading("[별표 4]에 해당하는 사업") is None
    lines = [
        L("제1조(목적)"),
        L("[별표 4]에 해당하는 사업은 제외한다."),
    ]
    units, _, _, _ = run(lines)
    assert all(u.part != "별표" for u in units)


def test_appendix_title_ignores_table_line_after_heading():
    table = Line(text="| 00. 00. | 000,000 |\n|---|---|", page=1, page_end=1, kind="table")
    lines = [
        L("제1조(목적)"),
        L("본문."),
        L("[별지 제6호서식]"),
        table,
    ]
    _, chunks, _, _ = run(lines)
    form = next(c for c in chunks if c.appendix_label == "별지 제6호서식")
    assert form.part == "별지"
    assert form.title == ""
    assert form.header == "[별지 제6호서식]"


def test_appendix_title_from_parenthesis_after_table_line():
    table = Line(text="| 구 비 서 류 | 1. 신청서 |", page=1, page_end=1, kind="table")
    lines = [
        L("제1조(목적)"),
        L("본문."),
        L("[별지 제6호서식] (제61조제1항 관련)"),
        table,
    ]
    _, chunks, _, _ = run(lines)
    form = next(c for c in chunks if c.appendix_label == "별지 제6호서식")
    assert form.title == "(제61조제1항 관련)"
    assert form.header == "[별지 제6호서식 (제61조제1항 관련)]"
    assert "|" not in form.title
