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



import streamlit as st

from dotenv import load_dotenv



load_dotenv()



INDEX_DIR = Path(__file__).resolve().parent / "vectorstore" / "faiss_index"



SUGGESTED_QUESTION_BUTTONS = [

    ("최저임금 확인", "2026년 적용 최저임금 시간급은 얼마인가요?"),

    ("휴게시간 확인", "8시간 근무하면 휴게시간은 얼마나 줘야 하나요?"),

    (
        "계약서 확인 항목",
        "표준근로계약서에서 확인해야 할 주요 항목은 무엇인가요?",
    ),

]

MENU_CHAT = "근로조건 상담"

MENU_PDF = "계약서·근무표 점검"






PDF_EXTRACT_FAIL_MSG = (

    "업로드한 PDF에서 텍스트를 추출하지 못했습니다. "

    "스캔본 또는 이미지형 PDF인지 확인해주세요."

)

PDF_SUPPORT_NOTICE = (

    "현재 텍스트 선택이 가능한 PDF 형식을 지원합니다. "

    "스캔본이나 이미지형 PDF는 텍스트 추출이 제한될 수 있습니다."

)


SOURCE_DISPLAY_NAMES = {

    "2026_최저임금_보도자료.txt": "고용노동부 2026년 최저임금 보도자료",

    "2025_개정_표준근로계약서.txt": "고용노동부 2025년 개정 표준근로계약서",

    "근로시간_휴게시간_FAQ.txt": "고용노동부 근로시간·휴게시간 FAQ",

}

CHUNK_META_SKIP_KEYWORDS = [

    "[처리]",

    "RAG용 텍스트 정리본",

    "BOM 제거",

    "탭 구분자 정리",

    "과도한 공백/빈 줄 정리",

]

DISPLAY_CHUNK_LIMIT = 3

DISPLAY_CHUNK_MAX_LEN = 700





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

    if "conversation_history" not in st.session_state:

        st.session_state["conversation_history"] = []

    if "history" not in st.session_state:

        st.session_state.history = []

    if "chat_question" not in st.session_state:

        st.session_state["chat_question"] = ""

    if "use_example_docs" not in st.session_state:

        st.session_state["use_example_docs"] = False





def _add_history(entry: dict) -> None:

    st.session_state.history.insert(0, entry)

    st.session_state.history = st.session_state.history[:10]


def _append_conversation_turn(
    question: str,
    llm_answer: str,
    rag_answer: str,
    contexts: list,
    sources: list,
) -> None:

    st.session_state["conversation_history"].append(
        {
            "user": question,
            "llm_answer": llm_answer,
            "rag_answer": rag_answer,
        }
    )

    _add_history(
        {
            "question": question,
            "llm_answer": llm_answer,
            "rag_answer": rag_answer,
            "retrieved_contexts": contexts,
            "sources": sources,
        }
    )





def _normalize_source_filename(source: str) -> str:

    return Path(source).name if source else source


def _friendly_source_name(source: str) -> str:

    filename = _normalize_source_filename(source)

    return SOURCE_DISPLAY_NAMES.get(filename, filename or source)


def _format_source_header(index: int, source: str) -> str:

    filename = _normalize_source_filename(source)

    friendly = _friendly_source_name(source)

    return f"**[{index}] {friendly}**\n\n출처 파일: {filename}"


def _clean_chunk_for_display(ctx: str, max_len: int = DISPLAY_CHUNK_MAX_LEN) -> str:

    lines = []

    for line in ctx.splitlines():

        if any(keyword in line for keyword in CHUNK_META_SKIP_KEYWORDS):

            continue

        lines.append(line)

    text = "\n".join(lines).strip()

    if len(text) > max_len:

        return text[:max_len] + "..."

    return text


def _format_sources_caption(sources: list) -> str:

    unique = list(dict.fromkeys(sources))

    labels = [_friendly_source_name(s) for s in unique]

    return ", ".join(labels)


def _show_source_info(label: str, source: str) -> None:

    from src.demo_check import get_source_display_message

    message = get_source_display_message(source)

    if source == "upload_failed":

        st.warning(f"{label}: {message}")

    elif source == "none":

        st.info(f"{label}: {message}")

    else:

        st.info(f"{label}: {message}")





def _render_chatbot_sidebar() -> None:

    st.subheader("최근 상담 기록")

    st.caption("최근 상담 문맥을 다음 질문에 반영합니다.")

    if st.button("상담 기록 초기화", key="reset_conversation"):

        st.session_state["conversation_history"] = []

        st.session_state.history = []

        if "chat_question" in st.session_state:

            del st.session_state["chat_question"]

        st.rerun()

    if not st.session_state.history:

        st.caption("아직 기록이 없습니다.")

    for i, item in enumerate(st.session_state.history):

        with st.expander(f"{i + 1}. {item['question'][:30]}...", expanded=False):

            st.markdown("**LLM**")

            st.write(item["llm_answer"][:200] + "...")

            st.markdown("**RAG**")

            st.write(item["rag_answer"][:200] + "...")


def page_chatbot(api_key: str) -> None:

    with st.sidebar:

        _render_chatbot_sidebar()

    st.header("근로조건 상담")

    st.caption(
        "같은 질문에 대해 문서 검색 없는 일반 LLM 답변과 "
        "공식 노동자료를 검색한 RAG 답변을 함께 제공합니다."
    )



    if not _index_ready():

        st.warning(

            "벡터 DB가 없습니다. 터미널에서 `python -m src.build_vector_db`를 먼저 실행하세요."

        )

        return



    os.environ["OPENAI_API_KEY"] = api_key



    st.markdown("**추천 질문**")

    demo_cols = st.columns(len(SUGGESTED_QUESTION_BUTTONS))

    for i, (label, question) in enumerate(SUGGESTED_QUESTION_BUTTONS):

        if demo_cols[i].button(label, use_container_width=True, key=f"demo_btn_{i}"):

            st.session_state["chat_question"] = question

    with st.expander("고급 검색 설정", expanded=False):

        top_k = st.slider("검색 문서 수", min_value=2, max_value=5, value=5)



    st.text_input(

        "질문을 입력하세요",

        key="chat_question",

        placeholder="예: 2026년 적용 최저임금 시간급은 얼마인가요?",

    )



    ask_clicked = st.button("질문하기", type="primary", key="ask_btn")



    if ask_clicked:

        q = st.session_state.get("chat_question", "").strip()

        if not q:

            st.warning("질문을 입력해주세요.")

        else:

            with st.spinner("순수 LLM과 RAG 답변을 생성 중..."):

                try:

                    from src.conversation_utils import (
                        build_llm_history,
                        build_rag_history,
                        verify_separated_history,
                    )

                    from src.llm_baseline import get_llm_answer

                    from src.rag_chain import get_rag_answer



                    previous_history = list(
                        st.session_state.get("conversation_history", [])
                    )

                    llm_history = build_llm_history(previous_history)

                    rag_history = build_rag_history(previous_history)

                    verify_separated_history(llm_history, rag_history)



                    llm_answer = get_llm_answer(question=q, history=llm_history)

                    rag_result = get_rag_answer(
                        question=q, top_k=top_k, history=rag_history
                    )

                    rag_answer = str(rag_result["answer"])

                    contexts = list(rag_result.get("contexts", []))

                    sources = list(rag_result.get("sources", []))



                    _append_conversation_turn(
                        q, llm_answer, rag_answer, contexts, sources
                    )

                except Exception as exc:

                    st.warning("답변 생성 중 오류가 발생했습니다. API Key와 네트워크 연결을 확인해주세요.")

                    return



            st.subheader(f"질문: {q}")

            col_llm, col_rag = st.columns(2)



            with col_llm:

                st.markdown("#### 순수 LLM 답변")

                st.caption("문서 검색 없이 생성한 답변입니다.")

                st.info(llm_answer)



            with col_rag:

                st.markdown("#### 공식자료 RAG 답변")

                st.caption("공식 노동자료를 검색하여 생성한 답변입니다.")

                st.success(rag_answer)



                with st.expander("검색된 공식 문서 보기", expanded=False):

                    display_pairs = list(

                        zip(contexts, sources or ["unknown"] * len(contexts))

                    )[:DISPLAY_CHUNK_LIMIT]

                    for i, (ctx, src) in enumerate(display_pairs, 1):

                        st.markdown(_format_source_header(i, src))

                        st.text(_clean_chunk_for_display(ctx))

                        if i < len(display_pairs):

                            st.divider()





def page_pdf_demo(api_key: str) -> None:

    st.header("계약서·근무표 점검")



    st.markdown(
        "근로계약서와 근무표를 업로드하면 주요 근로조건을 추출하고, "
        "공식 노동자료 기준과 비교하여 확인 결과를 제공합니다. "
        "업로드 문서는 공식 지식 DB에 저장하지 않고 현재 세션에서만 처리합니다."
    )

    st.caption(PDF_SUPPORT_NOTICE)

    st.markdown("**점검 항목**")

    st.markdown(
        """
- 계약서 시급이 2026년 최저임금 기준 이상인지 확인
- 근무시간 대비 휴게시간이 충분한지 확인
- 근로계약서 필수 항목 누락 여부 확인
- 근무표 입력 및 근무 배치 확인
"""
    )



    if not _index_ready():

        st.warning(

            "벡터 DB가 없습니다. 터미널에서 `python -m src.build_vector_db`를 먼저 실행하세요."

        )

        return



    os.environ["OPENAI_API_KEY"] = api_key



    from src.demo_check import (

        FALLBACK_CONTRACT,

        FALLBACK_SCHEDULE,

        RAG_EVIDENCE_QUESTIONS,

        SAMPLE_CONTRACT_PDF,

        SAMPLE_SCHEDULE_PDF,

        SHOW_DEV_SOURCE_INFO,

        check_break_time,

        check_minimum_wage,

        check_required_fields,

        check_schedule,

        format_field_status,

        resolve_example_document,

        resolve_uploaded_pdf,

    )



    col1, col2 = st.columns(2)

    with col1:

        contract_upload = st.file_uploader(

            "근로계약서 PDF 업로드",

            type=["pdf"],

            key="contract_upload",

        )

    with col2:

        schedule_upload = st.file_uploader(

            "근무표 PDF 업로드",

            type=["pdf"],

            key="schedule_upload",

        )



    if st.button("예시 문서로 체험하기", key="use_example_docs_btn"):

        st.session_state["use_example_docs"] = True

        st.rerun()



    if contract_upload is not None or schedule_upload is not None:

        st.session_state["use_example_docs"] = False



    if contract_upload is not None:

        contract_text, contract_source, contract_pdf_fail = resolve_uploaded_pdf(
            contract_upload
        )

    elif st.session_state.get("use_example_docs"):

        contract_text, contract_source, contract_pdf_fail = resolve_example_document(
            SAMPLE_CONTRACT_PDF, FALLBACK_CONTRACT
        )

    else:

        contract_text, contract_source, contract_pdf_fail = "", "none", False



    if schedule_upload is not None:

        schedule_text, schedule_source, schedule_pdf_fail = resolve_uploaded_pdf(
            schedule_upload
        )

    elif st.session_state.get("use_example_docs"):

        schedule_text, schedule_source, schedule_pdf_fail = resolve_example_document(
            SAMPLE_SCHEDULE_PDF, FALLBACK_SCHEDULE
        )

    else:

        schedule_text, schedule_source, schedule_pdf_fail = "", "none", False



    if contract_pdf_fail:

        st.warning(f"근로계약서: {PDF_EXTRACT_FAIL_MSG}")

    if schedule_pdf_fail:

        st.warning(f"근무표: {PDF_EXTRACT_FAIL_MSG}")



    with st.expander("계약서 내용 보기", expanded=False):

        if contract_text.strip():

            st.text(contract_text[:2000] + ("..." if len(contract_text) > 2000 else ""))

        else:

            st.caption("표시할 계약서 내용이 없습니다.")



    with st.expander("근무표 내용 보기", expanded=False):

        if schedule_text.strip():

            st.text(schedule_text[:2000] + ("..." if len(schedule_text) > 2000 else ""))

        else:

            st.caption("표시할 근무표 내용이 없습니다.")



    if st.button("점검하기", type="primary", key="run_pdf_check"):

        _show_source_info("근로계약서", contract_source)

        _show_source_info("근무표", schedule_source)

        if SHOW_DEV_SOURCE_INFO:

            st.caption(
                f"계약서 분석 소스: {contract_source} | "
                f"추출 텍스트 길이: {len(contract_text.strip())}자"
            )

            st.caption(
                f"근무표 분석 소스: {schedule_source} | "
                f"추출 텍스트 길이: {len(schedule_text.strip())}자"
            )



        st.subheader("점검 결과")



        st.markdown("#### A. 최저임금 점검")

        wage_result = check_minimum_wage(contract_text)

        st.markdown(f"- 계약서 시급: **{wage_result['contract_wage']}**")

        st.markdown(f"- 공식 기준: **{wage_result['official_standard']}**")

        st.markdown(f"- 결과: **{wage_result['result']}**")



        st.markdown("#### B. 휴게시간 점검")

        break_result = check_break_time(contract_text)

        st.markdown(f"- 근무시간: **{break_result['work_hours']}**")

        st.markdown(f"- 휴게시간: **{break_result['break_time']}**")

        st.markdown(f"- 전체 시간 구간: **{break_result['total_hours']}**")

        st.markdown(f"- 휴게시간 길이: **{break_result['break_duration']}**")

        st.markdown(f"- 실근로시간: **{break_result['actual_work_hours']}**")

        st.markdown(f"- 법정 최소 휴게시간: **{break_result['required_break']}**")

        st.markdown(f"- 결과: **{break_result['result']}**")

        if break_result.get("result_detail"):

            st.caption(break_result["result_detail"])



        st.markdown("#### C. 근로계약서 필수 항목 점검")

        for label, status in check_required_fields(contract_text):

            st.markdown(format_field_status(label, status))



        st.markdown("#### D. 근무표 확인")

        sched_result = check_schedule(schedule_text)

        st.markdown(f"- **{sched_result['status']}**")

        st.markdown(f"- {sched_result['detail']}")

        st.caption(sched_result["note"])



        st.subheader("공식 기준 RAG 근거")

        from src.rag_chain import get_rag_answer



        with st.spinner("공식 노동자료 RAG 검색 중..."):

            for q in RAG_EVIDENCE_QUESTIONS:

                st.markdown(f"**Q. {q}**")

                try:

                    rag = get_rag_answer(q, top_k=3)

                    st.success(str(rag["answer"]))

                    sources = rag.get("sources", [])

                    if sources:

                        st.caption("출처: " + _format_sources_caption(sources))

                except Exception as exc:

                    st.warning("RAG 검색을 완료하지 못했습니다. 잠시 후 다시 시도해주세요.")



        st.caption("본 결과는 공식 자료 기반 참고용 점검이며, 법률·노무 자문이 아닙니다.")



    st.divider()

    st.caption(

        "업로드한 계약서·근무표는 공식 지식 DB에 저장하지 않으며, "

        "현재 세션에서만 처리됩니다. "

    )





def main() -> None:

    st.set_page_config(

        page_title="근무체크",

        page_icon="📋",

        layout="wide",

        initial_sidebar_state="expanded",

    )



    _init_session_state()



    api_key = _load_api_key()



    with st.sidebar:

        st.title("근무체크")

        st.caption("공식 노동자료 기반 근로계약·근무조건 안내")

        page = st.radio(

            "메뉴",

            [MENU_CHAT, MENU_PDF],

            label_visibility="collapsed",

        )

        st.divider()

        st.markdown(

            """

            - 공식 노동자료 기반 답변
            - 법률·노무 자문이 아닌 참고 정보
            - 답변 근거와 출처 제공

            """

        )



    if not api_key:

        st.error(API_KEY_ERROR_MSG)

        st.stop()



    os.environ["OPENAI_API_KEY"] = api_key



    if page == MENU_PDF:

        page_pdf_demo(api_key)

    else:

        page_chatbot(api_key)





if __name__ == "__main__":

    main()


