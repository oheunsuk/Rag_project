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

from src.conversation_utils import format_model_history

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
1. 이전 RAG 대화에서 현재 질문의 대상을 파악하세요.
2. 구체적인 수치와 사실은 검색된 공식 문서 context를 근거로 답하세요.
3. context에 없는 내용은 추측하지 마세요.
4. 법률·노무 자문처럼 단정하지 마세요.
5. 답변은 짧고 명확하게 작성하세요.
6. 근로계약서 관련 질문에서는 일반 근로자에게 적용되는 서면 명시 항목과 기간제·단시간근로자에게 추가로 적용되는 항목을 가능한 범위에서 구분하세요.
7. 표준근로계약서 서식에 존재한다는 이유만으로 모든 항목을 모든 근로계약의 법정 필수 항목이라고 단정하지 마세요.
8. 구체적인 법률 해석은 검색된 공식 문서에 근거해서만 설명하세요.

근로계약서 주요 항목 질문 시 우선 구조:
- 일반적인 주요 서면 명시 항목: 임금의 구성항목·계산방법·지급방법, 소정근로시간, 휴일, 연차유급휴가
- 기간제·단시간근로자 추가 확인 항목: 근로계약기간, 근로시간과 휴게시간, 근무장소와 업무내용, 근로일과 근로일별 근로시간
- 표준 서식에는 사회보험 적용 여부와 계약서 교부 확인 등의 항목도 포함될 수 있음
검색 문서가 구분을 충분히 지원하지 않으면 억지로 법률 내용을 생성하지 말고, 검색된 표준근로계약서 기준의 주요 확인 항목임을 밝히고 근로형태에 따라 달라질 수 있음을 안내하세요."""

RAG_USER_PROMPT = """이전 대화:
{history}

현재 질문: {question}

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


def _extract_source_name(doc: Document) -> str:
    """문서 metadata 또는 본문에서 출처명 추출."""
    if doc.metadata.get("source_file"):
        return str(doc.metadata["source_file"])
    content = doc.page_content
    for line in content.splitlines()[:5]:
        if "[문서명]" in line:
            return line.split("[문서명]", 1)[-1].strip()
        if line.startswith("[문서:") and "]" in line:
            return line[1 : line.index("]")].replace("문서:", "").strip()
    return str(doc.metadata.get("source_path", "unknown"))


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

    def query(
        self,
        question: str,
        history: List[dict] | None = None,
    ) -> Tuple[str, List[str], List[str]]:
        """
        질문에 대한 RAG 답변, contexts, sources를 반환한다.

        Returns:
            (rag_answer, retrieved_contexts, sources)
        """
        docs = self.retrieve(question)
        contexts = _docs_to_context_strings(docs)
        sources = [_extract_source_name(doc) for doc in docs]
        context_text = _format_docs(docs)

        chain = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke(
            {
                "question": question,
                "context": context_text,
                "history": format_model_history(history),
            }
        )
        return answer, contexts, sources


def get_rag_answer(
    question: str,
    index_dir: Path | str | None = None,
    top_k: int = 5,
    history: List[dict] | None = None,
) -> Dict[str, object]:
    """
    RAG 답변 + 검색 context + 출처명 반환.

    Returns:
        answer   : RAG 답변
        contexts : 검색된 문서 chunk 리스트
        sources  : chunk별 문서명 리스트
    """
    chain = WorkcheckRAGChain(index_dir=index_dir, top_k=top_k)
    answer, contexts, sources = chain.query(question, history=history)
    return {"answer": answer, "contexts": contexts, "sources": sources}


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
