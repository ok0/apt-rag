"""OCR 엔진 인터페이스.

이미지 기반 쪽을 텍스트로 바꾸기 위한 엔진 후보는 다음과 같습니다.

- Apple Vision (ocrmac 또는 pyobjc-framework-Vision): macOS 전용, 추천.
  설치: uv add ocrmac
- Tesseract + kor: PyMuPDF get_textpage_ocr(language="kor")로 호출.
  설치: brew install tesseract tesseract-lang
- PaddleOCR korean (PP-Structure로 표 인식 가능).
  설치: uv add paddleocr paddlepaddle
- EasyOCR (ko).
  설치: uv add easyocr

현재는 "none"만 지원합니다. 엔진을 추가하면 get_ocr_engine에서 이름으로 연결합니다.
"""

from __future__ import annotations

from typing import Protocol


class OcrUnavailableError(RuntimeError):
    pass


class OcrEngine(Protocol):
    name: str

    def ocr_page(self, png_bytes: bytes, page_no: int) -> str: ...


class NullOcr:
    name = "none"

    def ocr_page(self, png_bytes: bytes, page_no: int) -> str:
        raise OcrUnavailableError(
            f"{page_no}쪽은 이미지 기반입니다. OCR 엔진이 설정되지 않았습니다(APT_RAG_OCR)."
        )


def get_ocr_engine(name: str) -> OcrEngine:
    if name == "none":
        return NullOcr()
    raise OcrUnavailableError(f"지원하지 않는 OCR 엔진: {name} (현재 none만 지원)")
