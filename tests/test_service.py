from pathlib import Path

import apt_rag.service as service
from apt_rag.models import ValidationReport
from apt_rag.service import PrepareResult, build_index


def test_build_index_reuses_prepare_sha256(monkeypatch, settings, hash_embedder, tmp_path):
    prep = PrepareResult(
        doc_id="doc",
        work_dir=tmp_path,
        page_count=1,
        sha256="abc123",
        extract=None,
        chunks=[],
        excluded=[],
        report=ValidationReport(summary={}, scopes={}, issues=[], has_errors=False),
    )
    monkeypatch.setattr(service, "prepare", lambda pdf, settings: prep)

    def fail_sha256(path: Path) -> str:
        raise AssertionError("build_index must not rehash the PDF")

    monkeypatch.setattr(service, "_sha256", fail_sha256)
    captured = {}
    monkeypatch.setattr(
        service, "write_index", lambda s, chunks, vectors, manifest: captured.update(manifest)
    )

    result = build_index(tmp_path / "x.pdf", settings, embedder=hash_embedder)

    assert result.indexed
    assert captured["sha256"] == "abc123"
