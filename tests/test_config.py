from apt_rag.config import load_settings


def test_synonyms_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv("APT_RAG_SYNONYMS", str(tmp_path / "syn.txt"))
    assert load_settings(tmp_path / ".env").synonyms_path == (tmp_path / "syn.txt").resolve()


def test_synonyms_path_default(monkeypatch, tmp_path):
    monkeypatch.delenv("APT_RAG_SYNONYMS", raising=False)
    assert load_settings(tmp_path / ".env").synonyms_path.name == "synonyms.txt"
