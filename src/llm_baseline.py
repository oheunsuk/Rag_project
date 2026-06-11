"""
순수 LLM baseline 답변 모듈 (Retriever/Vector DB 미사용).

실행 방법:
    python -m src.llm_baseline "2026년 최저임금은?"

환경변수:
    OPENAI_API_KEY
"""

from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.conversation_utils import format_model_history

load_dotenv()

DEFAULT_LLM_MODEL = "gpt-4o-mini"

LLM_BASELINE_SYSTEM = """당신은 근로계약·근무관리 관련 질문에 답하는 어시스턴트입니다.

답변 규칙:
1. 이전 대화 문맥을 참고하여 현재 질문에 답하세요.
2. 별도의 공식 문서 검색이나 외부 자료는 제공되지 않았습니다.
3. 확실하지 않은 최신 사실이나 수치는 추측하지 말고, 불확실성을 명확히 밝혀주세요.
4. 정보가 부족하면 필요한 추가 정보를 설명하세요.
5. 알고 있는 범위에서는 질문에 직접적이고 간결하게 답하세요.
6. 법적으로 반드시 서면 명시해야 하는 항목과 선택적으로 둘 수 있는 일반 계약 조항을 구분하여 설명하세요.
7. 확실하지 않은 법적 의무는 필수라고 단정하지 마세요."""

LLM_BASELINE_USER = """이전 대화:
{history}

현재 질문: {question}

위 질문에 답변하세요."""


def _get_llm(model: str = DEFAULT_LLM_MODEL) -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    return ChatOpenAI(model=model, temperature=0, openai_api_key=api_key)


def get_llm_answer(
    question: str,
    history: List[dict] | None = None,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> str:
    """
    Retriever 없이 순수 GPT 답변을 생성한다.

    Args:
        question: 사용자 질문
        history: 이전 순수 LLM 대화 기록 [{user, assistant}, ...]
        llm_model: OpenAI chat model 이름

    Returns:
        순수 LLM 답변 문자열
    """
    llm = _get_llm(model=llm_model)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", LLM_BASELINE_SYSTEM),
            ("human", LLM_BASELINE_USER),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "question": question,
            "history": format_model_history(history),
        }
    )


if __name__ == "__main__":
    import sys

    q = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "2026년 적용 최저임금 시간급은 얼마인가?"
    )
    print(get_llm_answer(q))
