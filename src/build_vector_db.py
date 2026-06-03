"""
OpenAI Embedding + FAISS 벡터 DB 생성 모듈.

실행 방법:
    python -m src.build_vector_db

환경변수:
    OPENAI_API_KEY
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.ingest_docs import ingest_documents

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / "vectorstore" / "faiss_index"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _get_embeddings(model: str = DEFAULT_EMBEDDING_MODEL) -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    return OpenAIEmbeddings(model=model, openai_api_key=api_key)


def build_vector_db(
    index_dir: Path | str | None = None,
    docs_dir: Path | str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> FAISS:
    """
    문서를 ingest하고 FAISS 벡터 DB를 생성·저장한다.

    Returns:
        생성된 FAISS vectorstore.
    """
    save_path = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
    save_path.mkdir(parents=True, exist_ok=True)

    chunks = ingest_documents(docs_dir=docs_dir)
    embeddings = _get_embeddings(model=embedding_model)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(save_path))

    print(f"FAISS 인덱스 저장 완료: {save_path}")
    print(f"총 {len(chunks)}개 chunk 벡터화")
    return vectorstore


if __name__ == "__main__":
    build_vector_db()
