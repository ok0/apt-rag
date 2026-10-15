# apt-rag

아파트 관리규약 PDF를 읽어 하이브리드(벡터 + BM25) 검색용 색인을 만드는 도구입니다.

## 1. 사전 설치

- uv만 필요합니다. tesseract는 불필요합니다. ollama는 평가 질문 생성(`gen-questions`)에만 필요합니다(5절 참고).
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
uv run apt-rag search <query> [--top-n 5] [--excerpt 160] [--json] [--min-sim 0.55]
uv run apt-rag eval <csv> [-k 5] [--json] [--min-sim 0.55]
uv run apt-rag gen-questions [--count 40] [--per-article 1] [--out eval/generated_questions.csv] [--model M] [--seed 42] [--parts 본문] [--min-chars 80] [--force]
```

- `inspect`: 쪽별 텍스트/이미지 판별 결과와 OCR 필요 쪽을 보여줍니다.
- `index`: PDF를 추출, 청크화, 검증한 뒤 색인을 만듭니다. `--no-embed`는 임베딩을 생략합니다. `--strict`는 검증 오류가 있으면 실패합니다.
- `search`: 색인에서 질의에 맞는 조항을 검색합니다. 결과의 `vec #N sim=0.xx`는 벡터 코사인 유사도입니다. `--min-sim`(기본 0.55, `APT_RAG_MIN_VEC_SIM`으로 변경 가능)보다 낮은 벡터 후보는 버리며, `0`이면 컷오프를 끕니다. 벡터 후보가 하나도 기준을 넘지 못하면 BM25 후보도 버리고 결과를 비워 `관련 조항을 찾지 못했습니다.`를 출력합니다. 벡터 후보가 하나라도 남으면 BM25와 RRF로 융합합니다. `0`이면 이 규칙도 함께 꺼지고 BM25 단독 결과도 나옵니다.
- `eval`: 질문 CSV로 hit@k 검색 적중률을 측정합니다. 컷오프로 결과가 비면 실패로 집계합니다. CSV는 UTF-8이며 `question,expected_article` 컬럼이 필요합니다. 정답 조문이 여러 개면 `;`로 구분하며, 상위 k개 중 하나라도 맞으면 적중입니다. 샘플은 `eval/sample_questions.csv`(15문항)이고, 실제 평가용 30~50문항은 같은 형식으로 작성합니다.

### 컷오프 기본값 실측 (`APT_RAG_MIN_VEC_SIM`, 기본 0.55)

예시 PDF를 BGE-M3로 색인한 뒤, 질문별 벡터 top1 코사인 유사도를 쟀습니다.

| 질문 집합 | 개수 | 최소 | 중앙값 | 최대 |
|---|---|---|---|---|
| 규약 관련 (`eval/sample_questions.csv`) | 15 | 0.607 | 0.695 | 0.795 |
| 규약 무관 (일상 질문) | 15 | 0.337 | 0.426 | 0.518 |

- 두 집합은 0.518~0.607 사이에서 갈립니다. 0.55는 이 구간의 중간 근처입니다.
- `eval --min-sim`별 hit@5: `0` 15/15, `0.55` 15/15, `0.6` 15/15, `0.65` 14/15.
- 무관 질문 15개(위 표)는 기본값에서 전부 0개 결과입니다. 벡터 후보가 전부 컷오프에 걸리면 BM25 후보도 버리기 때문입니다.

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

## 5. 평가 질문 생성

색인된 조항을 로컬 LLM(ollama)에 보여주고, 입주민이 물어볼 법한 질문과 정답 조항이 채워진 CSV 초안을 만듭니다. 직접 질문을 쓰는 부담을 줄이기 위한 기능입니다.

1. ollama 설치 및 서버 실행

   ```bash
   brew install ollama
   brew services start ollama   # 또는 별도 터미널에서 `ollama serve`
   ```

2. 모델 받기 (`qwen2.5:7b`, 약 4.7GB)

   ```bash
   ollama pull qwen2.5:7b
   ```

3. 질문 생성

   ```bash
   uv run apt-rag gen-questions --count 40
   uv run apt-rag gen-questions --count 30 --parts 본문,별표 --out eval/generated_questions.csv
   ```

- 색인(`apt-rag index`)이 먼저 있어야 합니다. 없으면 종료됩니다.
- 출력은 `question,expected_article,source_title,source_pages` 열을 가진 CSV이며, `apt-rag eval`에 그대로 넘길 수 있습니다.
- `expected_article`은 생성기가 질문을 만든 조항 번호입니다(예: `제69조`, `별표 8`).
- 서버 주소와 모델은 `APT_RAG_OLLAMA_URL`, `APT_RAG_LLM_MODEL` 환경변수(또는 `.env`)로 바꿀 수 있고, `--model`이 우선합니다.
- 생성 결과는 초안입니다. 정답 조항이 자동으로 붙어 있어도 사람이 질문과 정답을 반드시 검토한 뒤 평가에 사용하세요.
- 검색 결과를 보고 정답을 정하면 검색이 잘 되는 쪽으로 평가가 후하게 나옵니다. 생성기의 정답은 조 본문에서 만든 질문이므로 이런 편향이 적습니다.
