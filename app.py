"""
근무체크 RAG Streamlit 앱 (발표/배포용).

실행 방법:
    streamlit run app.py

환경변수:
    OPENAI_API_KEY  (.env, 시스템 환경변수, 또는 Streamlit secrets)
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
INDEX_DIR = PROJECT_ROOT / "vectorstore" / "faiss_index"

DEMO_QUESTIONS = [
    "2026년 적용 최저임금 시간급은 얼마인가?",
    "근로계약서에 반드시 포함되어야 하는 항목은?",
    "8시간 이상 근무할 때 필요한 휴게시간은?",
]

DEFAULT_RAGAS = {
    "Faithfulness": 0.884,
    "Answer Relevancy": 0.852,
    "Context Precision": 0.845,
    "Context Recall": 1.000,
    "Average": 0.895,
}

DEFAULT_TOPK = [
    {"top_k": 2, "average": 0.815},
    {"top_k": 3, "average": 0.857},
    {"top_k": 5, "average": 0.891},
]


def _load_api_key() -> str | None:
    """API Key 로드 순서: 1) Streamlit secrets 2) 환경변수 3) .env"""
    try:
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            return str(st.secrets["OPENAI_API_KEY"])
    except Exception:
        pass
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")


API_KEY_ERROR_MSG = (
    "OpenAI API Key가 설정되지 않았습니다. "
    "로컬에서는 .env 파일을, Streamlit Cloud에서는 Secrets 설정을 확인해주세요."
)


def _index_ready() -> bool:
    return (INDEX_DIR / "index.faiss").exists() and (INDEX_DIR / "index.pkl").exists()


def _init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def _add_history(entry: dict) -> None:
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:10]


def _load_csv_safe(path: Path) -> pd.DataFrame | None:
    if path.exists():
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return None
    return None


def _load_ragas_results() -> pd.DataFrame:
    df = _load_csv_safe(OUTPUTS_DIR / "ragas_results.csv")
    if df is not None and not df.empty:
        return df
    return pd.DataFrame(
        [{"metric": k, "score": v} for k, v in DEFAULT_RAGAS.items() if k != "Average"]
    )


def _load_topk_comparison() -> pd.DataFrame:
    df = _load_csv_safe(OUTPUTS_DIR / "topk_comparison.csv")
    if df is not None and not df.empty:
        return df
    return pd.DataFrame(DEFAULT_TOPK)


def _show_image_safe(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"`{path.name}` 파일이 없습니다. `python -m src.run_ragas_eval`로 생성할 수 있습니다.")


def page_chatbot(api_key: str) -> None:
    st.header("챗봇: 순수 LLM vs RAG 비교")
    st.caption("같은 질문에 대해 문서 검색 없는 GPT 답변과 공식 노동자료 RAG 답변을 나란히 비교합니다.")

    if not _index_ready():
        st.warning(
            "벡터 DB가 없습니다. 터미널에서 `python -m src.build_vector_db`를 먼저 실행하세요."
        )
        return

    os.environ["OPENAI_API_KEY"] = api_key

    col_demo, col_topk = st.columns([3, 1])
    with col_demo:
        st.markdown("**데모 질문 예시**")
        demo_cols = st.columns(len(DEMO_QUESTIONS))
        demo_q = None
        for i, q in enumerate(DEMO_QUESTIONS):
            if demo_cols[i].button(f"예시 {i + 1}", use_container_width=True):
                demo_q = q
    with col_topk:
        top_k = st.slider("RAG top_k", min_value=2, max_value=5, value=5)

    default_q = demo_q or ""
    question = st.text_input(
        "질문을 입력하세요",
        value=default_q,
        placeholder="예: 2026년 적용 최저임금 시간급은 얼마인가?",
    )

    if st.button("질문하기", type="primary") and question.strip():
        q = question.strip()
        with st.spinner("순수 LLM과 RAG 답변을 생성 중..."):
            try:
                from src.llm_baseline import get_llm_answer
                from src.rag_chain import get_rag_answer

                llm_answer = get_llm_answer(q)
                rag_result = get_rag_answer(q, top_k=top_k)
                rag_answer = str(rag_result["answer"])
                contexts = list(rag_result.get("contexts", []))
                sources = list(rag_result.get("sources", []))

                _add_history(
                    {
                        "question": q,
                        "llm_answer": llm_answer,
                        "rag_answer": rag_answer,
                        "retrieved_contexts": contexts,
                        "sources": sources,
                    }
                )
            except Exception as exc:
                st.error(f"오류가 발생했습니다: {exc}")
                return

        st.subheader(f"질문: {q}")
        col_llm, col_rag = st.columns(2)

        with col_llm:
            st.markdown("#### 순수 LLM 답변")
            st.caption("문서 검색 없이 GPT만 사용한 답변")
            st.info(llm_answer)

        with col_rag:
            st.markdown("#### RAG 답변")
            st.caption("공식 노동자료 검색 기반 답변")
            st.success(rag_answer)

            with st.expander("검색된 공식 문서 보기", expanded=False):
                for i, (ctx, src) in enumerate(
                    zip(contexts, sources or ["unknown"] * len(contexts)), 1
                ):
                    st.markdown(f"**[{i}] {src}**")
                    preview = ctx[:600] + ("..." if len(ctx) > 600 else "")
                    st.text(preview)
                    if i < len(contexts):
                        st.divider()

    with st.sidebar:
        st.subheader("이전 질문 기록")
        if not st.session_state.history:
            st.caption("아직 기록이 없습니다.")
        for i, item in enumerate(st.session_state.history):
            with st.expander(f"{i + 1}. {item['question'][:30]}...", expanded=False):
                st.markdown("**LLM**")
                st.write(item["llm_answer"][:200] + "...")
                st.markdown("**RAG**")
                st.write(item["rag_answer"][:200] + "...")


def page_evaluation() -> None:
    st.header("성능 평가")
    st.caption("RAGAS 정량 평가 및 Top-k 비교 실험 결과")

    ragas_df = _load_ragas_results()
    st.subheader("RAGAS 4지표 결과")
    st.dataframe(ragas_df, use_container_width=True, hide_index=True)

    avg = ragas_df["score"].mean()
    if (OUTPUTS_DIR / "ragas_results.csv").exists():
        st.metric("평균 점수", f"{avg:.3f}")
    else:
        st.metric("평균 점수 (기본값)", f"{DEFAULT_RAGAS['Average']:.3f}")

    col1, col2 = st.columns(2)
    with col1:
        _show_image_safe(OUTPUTS_DIR / "ragas_chart.png", "RAGAS 지표 막대그래프")
    with col2:
        _show_image_safe(
            OUTPUTS_DIR / "topk_comparison_chart.png", "Top-k 비교 막대그래프"
        )

    st.subheader("Top-k 비교 결과")
    topk_df = _load_topk_comparison()
    display_cols = [c for c in topk_df.columns if c in topk_df.columns]
    st.dataframe(topk_df[display_cols], use_container_width=True, hide_index=True)

    if not (OUTPUTS_DIR / "topk_comparison.csv").exists():
        st.caption("Top-k CSV가 없어 기본 실험값을 표시합니다.")

    comparison_path = OUTPUTS_DIR / "llm_vs_rag_comparison.csv"
    if comparison_path.exists():
        st.subheader("순수 LLM vs RAG 비교 (실험 데이터)")
        cmp_df = pd.read_csv(comparison_path, encoding="utf-8-sig")
        st.dataframe(
            cmp_df[
                [
                    "id",
                    "question",
                    "llm_has_source",
                    "rag_has_source",
                    "llm_answer_length",
                    "rag_answer_length",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "`outputs/llm_vs_rag_comparison.csv` 파일이 없습니다. "
            "로컬에서 비교 실험 스크립트 실행 후 결과를 포함해 주세요."
        )


def page_info() -> None:
    st.header("프로젝트 정보")

    st.subheader("서비스 개요")
    st.markdown(
        """
**근무체크 RAG**는 고용노동부 공식 노동자료를 기반으로
근로계약·근무조건 관련 질문에 답하는 RAG 서비스입니다.
"""
    )

    st.subheader("서비스 대상")
    st.markdown(
        """
- 아르바이트 근로자
- 소규모 사업주
- 근무표·시급 관리 담당자
"""
    )

    st.subheader("사용 문서")
    st.markdown(
        """
- 2025년 개정 표준근로계약서
- 2026년 적용 최저임금 보도자료
- 근로시간·휴게시간 FAQ
- 근로계약서 작성방법 안내
"""
    )

    st.subheader("기술 스택")
    st.markdown(
        """
| 구분 | 기술 |
|------|------|
| UI | Streamlit |
| RAG | LangChain |
| Vector DB | FAISS |
| Embedding | OpenAI text-embedding-3-small |
| LLM | GPT-4o-mini |
| 평가 | RAGAS |
"""
    )

    st.subheader("핵심 차별점")
    st.markdown(
        """
1. **순수 LLM vs RAG** 답변을 한 화면에서 비교
2. **공식 문서 기반** 근거(chunk·문서명) 확인
3. **RAGAS** 4지표 정량 평가 (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
4. **Top-k** 검색 개수별 성능 비교 실험
"""
    )

    st.subheader("링크")
    st.markdown("**GitHub URL:** TODO")
    st.markdown("**배포 URL:** TODO")

    st.subheader("발표 데모용 질문 예시")
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        st.markdown(f"{i}. {q}")


def main() -> None:
    st.set_page_config(
        page_title="근무체크 RAG",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session_state()

    api_key = _load_api_key()

    with st.sidebar:
        st.title("근무체크 RAG")
        st.caption("공식 노동자료 기반 근로계약·근무조건 안내")
        page = st.radio(
            "메뉴",
            ["챗봇", "성능 평가", "프로젝트 정보"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown(
            """
            - 공식 자료만 근거로 답변
            - 법률·노무 **자문 아님**
            - 참고용 정보 제공
            """
        )

    if page == "성능 평가":
        page_evaluation()
        return

    if page == "프로젝트 정보":
        page_info()
        return

    if not api_key:
        st.error(API_KEY_ERROR_MSG)
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key
    page_chatbot(api_key)


if __name__ == "__main__":
    main()
