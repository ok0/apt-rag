from apt_rag.models import page_range


def test_page_range_single_page_default_prefix():
    assert page_range(3, 3) == "p.3"


def test_page_range_span_default_prefix():
    assert page_range(3, 5) == "p.3-5"


def test_page_range_without_prefix():
    assert page_range(7, 7, prefix="") == "7"
    assert page_range(7, 9, prefix="") == "7-9"
