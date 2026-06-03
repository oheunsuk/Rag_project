"""
공식 노동자료(PDF/TXT) 로드 및 chunk 분할 모듈.

실행 방법:
    python -m src.ingest_docs
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_DIR = PROJECT_ROOT / "data" / "official_docs"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


def _load_single_file(file_path: Path) -> List[Document]:
    """단일 PDF 또는 TXT 파일을 로드한다."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
    elif suffix == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
    else:
        return []

    for doc in docs:
        doc.metadata["source_file"] = file_path.name
        doc.metadata["source_path"] = str(file_path.relative_to(PROJECT_ROOT))

    return docs


def load_documents(docs_dir: Path | str | None = None) -> List[Document]:
    """
    official_docs 폴더 내 PDF/TXT 문서를 재귀적으로 로드한다.

    Args:
        docs_dir: 문서 루트 디렉터리. 기본값은 data/official_docs.

    Returns:
        로드된 Document 리스트.
    """
    root = Path(docs_dir) if docs_dir else DEFAULT_DOCS_DIR
    if not root.exists():
        raise FileNotFoundError(f"문서 디렉터리를 찾을 수 없습니다: {root}")

    documents: List[Document] = []
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in {".pdf", ".txt"}:
            documents.extend(_load_single_file(file_path))

    if not documents:
        raise ValueError(f"로드된 문서가 없습니다. PDF/TXT 파일을 확인하세요: {root}")

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """RecursiveCharacterTextSplitter로 문서를 chunk 분할한다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_documents(
    docs_dir: Path | str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """문서 로드 후 chunk 분할까지 수행한다."""
    documents = load_documents(docs_dir)
    return split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


if __name__ == "__main__":
    chunks = ingest_documents()
    print(f"총 {len(chunks)}개 chunk 생성")
    for i, chunk in enumerate(chunks[:3]):
        source = chunk.metadata.get("source_file", "unknown")
        print(f"\n--- chunk {i + 1} ({source}) ---")
        print(chunk.page_content[:200])
