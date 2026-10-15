# apt-rag

아파트 관리규약 PDF를 읽어 하이브리드(벡터 + BM25) 검색용 색인을 만드는 도구입니다.

## 1. 사전 설치

- uv만 필요합니다. ollama, tesseract는 불필요합니다.
- Python 3.12는 uv가 자동으로 설치합니다.
- 임베딩 모델 BGE-M3(약 2.3GB)는 첫 실행 시 다운로드됩니다.
- 네트워크가 필요한 시점은 두 가지뿐입니다.
  - 첫 `uv sync` (PyPI)
  - 첫 모델 로드 (huggingface.co)
- 이후에는 `HF_HUB_OFFLINE=1` 설정으로 오프라인 실행이 가능합니다.

```bash
uv sync
```

## 2. 명령 사용법

```bash
uv run apt-rag inspect <pdf> [--json]
uv run apt-rag index <pdf> [--no-embed] [--strict]
uv run apt-rag search <query> [--top-n 5] [--excerpt 160] [--json]
uv run apt-rag eval <csv> [-k 5] [--json]
```

- `inspect`: 쪽별 텍스트/이미지 판별 결과와 OCR 필요 쪽을 보여줍니다.
- `index`: PDF를 추출, 청크화, 검증한 뒤 색인을 만듭니다. `--no-embed`는 임베딩을 생략합니다. `--strict`는 검증 오류가 있으면 실패합니다.
- `search`: 색인에서 질의에 맞는 조항을 검색합니다.
- `eval`: 질문 CSV로 검색 적중률을 측정합니다.

## 3. 산출물 위치

산출물은 `APT_RAG_DATA_DIR`(기본값 `data`) 아래에 저장됩니다.

- `data/work/<doc_id>/`
  - `inspect.json`: 쪽별 판별 결과
  - `pages.json`: 쪽 정보, 제거된 줄, 통계
  - `extracted.md`: 추출된 본문(쪽 마커 포함)
  - `chunks.json`: 청크, 제외 구간, 구조 줄
  - `validation_report.json`, `validation_report.md`: 검증 리포트
- `data/index/`: 색인 (`chroma/`, `chunks.json`, `bm25_tokens.json`, `manifest.json`)
- `data/work/eval/eval-YYYYmmdd-HHMMSS.json`: 평가 결과

설정은 `.env.example`을 참고하세요.

## 4. 테스트

```bash
uv run pytest
uv run pytest -m slow
```

- `uv run pytest`: 빠른 테스트만 실행합니다(실제 모델 미사용).
- `uv run pytest -m slow`: 실제 BGE-M3 모델을 다운로드/로드하는 테스트를 실행합니다.
