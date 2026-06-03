"""
FAISS 기반 RAG 체인 모듈.

실행 방법:
    python -m src.rag_chain "2026년 최저임금은?"
    python -m src.rag_chain "2026년 최저임금은?" --top_k 3

환경변수:
    OPENAI_API_KEY
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = PROJECT_ROOT / "vectorstore" / "faiss_index"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_TOP_K = 4

RAG_SYSTEM_PROMPT = """당신은 공식 노동자료 기반 근무체크 RAG 어시스턴트입니다.

답변 규칙:
1. 반드시 아래 제공된 공식 문서 context만 기반으로 답변하세요.
2. 문서에 없는 내용은 "문서에서 확인할 수 없습니다"라고 답변하세요.
3. 법률·노무 자문처럼 단정하지 마세요.
4. "공식 자료 기준으로 확인해야 할 항목"이라는 표현을 사용하세요.
5. 답변에는 가능하면 근거 문서명(source_file)을 포함하세요."""

RAG_USER_PROMPT = """질문: {question}

검색된 공식 문서:
{context}

위 문서만 근거로 답변하세요."""


def _get_embeddings(model: str = DEFAULT_EMBEDDING_MODEL) -> OpenAIEmbeddings:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    return OpenAIEmbeddings(model=model, openai_api_key=api_key)


def _get_llm(model: str = DEFAULT_LLM_MODEL) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    return ChatOpenAI(model=model, temperature=0, openai_api_key=api_key)


def load_vectorstore(index_dir: Path | str | None = None) -> FAISS:
    """저장된 FAISS 인덱스를 로드한다."""
    path = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
    index_files = (path / "index.faiss").exists() and (path / "index.pkl").exists()
    if not index_files:
        raise FileNotFoundError(
            f"FAISS 인덱스를 찾을 수 없습니다: {path}\n"
            "먼저 python -m src.build_vector_db 를 실행하세요."
        )
    embeddings = _get_embeddings()
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_retriever(
    vectorstore: FAISS | None = None,
    top_k: int = DEFAULT_TOP_K,
):
    """FAISS retriever를 생성한다."""
    store = vectorstore or load_vectorstore()
    return store.as_retriever(search_kwargs={"k": top_k})


def _format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        parts.append(f"[문서: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _docs_to_context_strings(docs: List[Document]) -> List[str]:
    return [doc.page_content for doc in docs]


class WorkcheckRAGChain:
    """근무체크 RAG 체인."""

    def __init__(
        self,
        index_dir: Path | str | None = None,
        top_k: int = DEFAULT_TOP_K,
        llm_model: str = DEFAULT_LLM_MODEL,
    ):
        self.vectorstore = load_vectorstore(index_dir)
        self.retriever = get_retriever(self.vectorstore, top_k=top_k)
        self.llm = _get_llm(model=llm_model)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_SYSTEM_PROMPT),
                ("human", RAG_USER_PROMPT),
            ]
        )

    def retrieve(self, question: str) -> List[Document]:
        return self.retriever.invoke(question)

    def query(self, question: str) -> Tuple[str, List[str]]:
        """
        질문에 대한 RAG 답변과 retrieved_contexts를 반환한다.

        Returns:
            (rag_answer, retrieved_contexts)
        """
        docs = self.retrieve(question)
        contexts = _docs_to_context_strings(docs)
        context_text = _format_docs(docs)

        chain = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"question": question, "context": context_text})
        return answer, contexts


def get_rag_answer(
    question: str,
    index_dir: Path | str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, object]:
    """
    ragas_guide.py get_rag_answer()와 동일한 반환 형식.

    Returns:
        answer   : LLM이 생성한 답변
        contexts : 검색된 문서 조각들 (리스트)
    """
    chain = WorkcheckRAGChain(index_dir=index_dir, top_k=top_k)
    answer, contexts = chain.query(question)
    return {"answer": answer, "contexts": contexts}


def query_rag(
    question: str,
    index_dir: Path | str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> Dict[str, object]:
    """단일 질문 RAG 실행 헬퍼."""
    result = get_rag_answer(question, index_dir=index_dir, top_k=top_k)
    return {
        "question": question,
        "answer": result["answer"],
        "contexts": result["contexts"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Workcheck RAG 단일 질의")
    parser.add_argument(
        "question",
        nargs="?",
        default="2026년 적용 최저임금 시간급은 얼마인가?",
        help="질문 문자열",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Retriever 검색 chunk 수 (기본값: {DEFAULT_TOP_K})",
    )
    args = parser.parse_args()

    result = query_rag(args.question, top_k=args.top_k)
    print("질문:", result["question"])
    print(f"top_k: {args.top_k}")
    print("\n답변:", result["answer"])
    print(f"\n검색 chunk 수: {len(result['contexts'])}")
