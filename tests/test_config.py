from apt_rag.config import load_settings


def test_synonyms_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv("APT_RAG_SYNONYMS", str(tmp_path / "syn.txt"))
    assert load_settings(tmp_path / ".env").synonyms_path == (tmp_path / "syn.txt").resolve()


def test_synonyms_path_default(monkeypatch, tmp_path):
    monkeypatch.delenv("APT_RAG_SYNONYMS", raising=False)
    assert load_settings(tmp_path / ".env").synonyms_path.name == "synonyms.txt"


def test_band_env(monkeypatch, tmp_path):
    monkeypatch.setenv("APT_RAG_SIM_BAND", "0.03")
    monkeypatch.setenv("APT_RAG_BAND_TOP_K", "7")
    s = load_settings(tmp_path / ".env")
    assert s.sim_band == 0.03
    assert s.band_top_k == 7
    monkeypatch.delenv("APT_RAG_SIM_BAND", raising=False)
    monkeypatch.delenv("APT_RAG_BAND_TOP_K", raising=False)
    s = load_settings(tmp_path / ".env")
    assert s.sim_band == 0.05
    assert s.band_top_k == 20


def test_llm_env(monkeypatch, tmp_path):
    monkeypatch.setenv("APT_RAG_LLM", "claude")
    s = load_settings(tmp_path / ".env")
    assert s.llm_provider == "claude"
    monkeypatch.delenv("APT_RAG_LLM", raising=False)
    s = load_settings(tmp_path / ".env")
    assert s.llm_provider == "ollama"
