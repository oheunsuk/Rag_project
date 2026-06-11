"""챗봇 공통 대화 기록 포맷 유틸."""

from __future__ import annotations

from typing import List

MAX_HISTORY_TURNS = 4


def build_llm_history(
    conversation_history: List[dict],
    max_turns: int = MAX_HISTORY_TURNS,
) -> List[dict]:
    recent_turns = conversation_history[-max_turns:]
    return [
        {
            "user": turn.get("user", ""),
            "assistant": turn.get("llm_answer", ""),
        }
        for turn in recent_turns
        if turn.get("user")
    ]


def build_rag_history(
    conversation_history: List[dict],
    max_turns: int = MAX_HISTORY_TURNS,
) -> List[dict]:
    recent_turns = conversation_history[-max_turns:]
    return [
        {
            "user": turn.get("user", ""),
            "assistant": turn.get("rag_answer", ""),
        }
        for turn in recent_turns
        if turn.get("user")
    ]


def format_model_history(
    history: List[dict] | None,
    max_turns: int = MAX_HISTORY_TURNS,
) -> str:
    """모델별 history(user/assistant)를 프롬프트용 텍스트로 변환한다."""
    if not history:
        return "(이전 대화 없음)"

    lines: List[str] = []
    for turn in history[-max_turns:]:
        user = turn.get("user", "")
        assistant = turn.get("assistant", "")
        if user:
            lines.append(f"사용자: {user}")
        if assistant:
            lines.append(f"도우미: {assistant}")
        lines.append("")

    return "\n".join(lines).strip()


def verify_separated_history(
    llm_history: List[dict],
    rag_history: List[dict],
) -> None:
    """개발 검증: 모델별 history에 교차 답변이 포함되지 않았는지 확인."""
    assert all("rag_answer" not in item for item in llm_history)
    assert all("llm_answer" not in item for item in rag_history)
    assert all("assistant" in item for item in llm_history)
    assert all("assistant" in item for item in rag_history)
