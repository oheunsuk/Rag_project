"""
PDF 텍스트 추출 유틸 (pypdf 기반, OCR 미사용).

실행 방법:
    python -m src.pdf_utils path/to/file.pdf
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

MIN_TEXT_LENGTH = 30

PdfInput = Union[str, Path, bytes, BinaryIO]


def extract_text_from_pdf(file_or_path: PdfInput) -> str:
    """
    pypdf로 PDF 텍스트를 추출한다.

    Args:
        file_or_path: 파일 경로, bytes, 또는 file-like 객체

    Returns:
        추출된 텍스트. 실패·빈 텍스트 시 빈 문자열.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    try:
        if isinstance(file_or_path, (str, Path)):
            reader = PdfReader(str(file_or_path))
        elif isinstance(file_or_path, bytes):
            reader = PdfReader(io.BytesIO(file_or_path))
        else:
            reader = PdfReader(file_or_path)

        parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(text)

        result = "\n".join(parts).strip()
        if len(result) < MIN_TEXT_LENGTH:
            return ""
        return result
    except Exception:
        return ""


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.pdf_utils <pdf_path>")
        sys.exit(1)
    text = extract_text_from_pdf(sys.argv[1])
    print(text[:500] if text else "(empty)")
