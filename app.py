"""
근무체크 RAG Streamlit 앱.

실행 방법:
    streamlit run app.py

환경변수:
    OPENAI_API_KEY
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_DIR = PROJECT_ROOT / "vectorstore" / "faiss_index"


def _index_ready() -> bool:
    return (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "index.pkl").exists()


@st.cache_resource
def get_rag_chain():
    from src.rag_chain import WorkcheckRAGChain

    return WorkcheckRAGChain()


def main():
    st.set_page_config(
        page_title="근무체크 RAG",
        page_icon="📋",
        layout="wide",
    )

    st.title("근무체크: 공식 노동자료 RAG")
    st.caption("아르바이트 근로자 · 소규모 매장 점주 · 근무표·시급 관리 담당자를 위한 서비스")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY 환경변수를 설정해 주세요.")
        st.stop()

    if not _index_ready():
        st.warning(
            "벡터 DB가 아직 생성되지 않았습니다. "
            "터미널에서 `python -m src.build_vector_db`를 먼저 실행하세요."
        )
        st.stop()

    with st.sidebar:
        st.header("안내")
        st.markdown(
            """
            - **공식 노동자료**만 근거로 답변합니다.
            - 법률·노무 **자문이 아닙니다**.
            - 답변은 참고용이며, **공식 자료 기준으로 확인**해야 합니다.
            """
        )
        st.divider()
        st.markdown("**사용 문서**")
        st.markdown(
            """
            - 2025 개정 표준근로계약서
            - 근로계약서 작성방법 안내
            - 2026년 적용 최저임금 자료
            - 근로시간·휴게시간 FAQ
            """
        )

    question = st.text_input(
        "질문을 입력하세요",
        placeholder="예: 2026년 적용 최저임금 시간급은 얼마인가?",
    )

    if st.button("질문하기", type="primary") and question.strip():
        with st.spinner("공식 자료를 검색하고 답변을 생성 중..."):
            try:
                rag = get_rag_chain()
                answer, contexts = rag.query(question.strip())
            except Exception as exc:
                st.error(f"오류가 발생했습니다: {exc}")
                st.stop()

        st.subheader("답변")
        st.info(answer)

        with st.expander("검색된 문서 context", expanded=False):
            for i, ctx in enumerate(contexts, 1):
                st.markdown(f"**Chunk {i}**")
                st.text(ctx[:800] + ("..." if len(ctx) > 800 else ""))


if __name__ == "__main__":
    main()
