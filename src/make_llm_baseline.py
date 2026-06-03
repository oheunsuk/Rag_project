"""
순수 LLM vs RAG 답변 비교 실험 모듈.

Retriever/Vector DB 없이 GPT만으로 baseline 답변을 생성하고,
기존 eval_dataset.csv의 RAG 답변과 병합·비교합니다.

실행 방법:
    python -m src.make_llm_baseline

사전 조건:
    outputs/eval_dataset.csv (python -m src.make_eval_dataset 실행 후)

환경변수:
    OPENAI_API_KEY  (.env 또는 시스템 환경변수)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.make_eval_dataset import load_ground_truth

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LLM_BASELINE = PROJECT_ROOT / "outputs" / "llm_baseline_answers.csv"
DEFAULT_EVAL_DATASET = PROJECT_ROOT / "outputs" / "eval_dataset.csv"
DEFAULT_COMPARISON = PROJECT_ROOT / "outputs" / "llm_vs_rag_comparison.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "outputs" / "llm_vs_rag_summary.txt"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

LLM_BASELINE_SYSTEM = """당신은 근로계약·근무관리 관련 질문에 답하는 어시스턴트입니다.

답변 규칙:
1. 일반적인 GPT 답변처럼 자연스럽게 답변하세요.
2. 별도의 공식 문서 context는 제공되지 않습니다.
3. 최신 법령·고시·공식 자료의 구체적 수치·시행일은 단정하지 마세요.
4. 확실하지 않은 최신 정보는 "최신 공식 자료 확인이 필요합니다"라고 안내하세요."""

LLM_BASELINE_USER = """질문: {question}

위 질문에 답변하세요."""

SOURCE_KEYWORDS = [
    "고용노동부",
    "공식",
    "문서",
    "자료",
    "출처",
    "근거",
    "기준",
    "faq",
    "표준근로계약서",
    "작성방법",
    "최저임금",
    "근로시간",
    "[문서:",
    "source_file",
    "공식 자료",
]


def _get_llm(model: str = DEFAULT_LLM_MODEL) -> ChatOpenAI:
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    return ChatOpenAI(model=model, temperature=0, openai_api_key=api_key)


def generate_llm_answer(question: str, llm: ChatOpenAI | None = None) -> str:
    """Retriever 없이 순수 LLM 답변 생성."""
    model = llm or _get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", LLM_BASELINE_SYSTEM),
            ("human", LLM_BASELINE_USER),
        ]
    )
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"question": question})


def has_source_expression(text: str) -> bool:
    """답변에 출처/근거 표현이 포함되어 있는지 판별."""
    if pd.isna(text) or not str(text).strip():
        return False
    lowered = str(text).lower()
    return any(keyword.lower() in lowered for keyword in SOURCE_KEYWORDS)


def build_llm_baseline(
    ground_truth_path: Path | str | None = None,
    output_path: Path | str | None = None,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> pd.DataFrame:
    """ground_truth.csv 기반 순수 LLM baseline 답변 생성."""
    gt_df = load_ground_truth(ground_truth_path)
    out_path = Path(output_path) if output_path else DEFAULT_LLM_BASELINE
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm = _get_llm(model=llm_model)
    rows = []

    print("\n🔄 순수 LLM baseline 답변 생성 중...")
    for _, row in gt_df.iterrows():
        question = str(row["question"])
        answer = generate_llm_answer(question, llm=llm)
        rows.append(
            {
                "id": row["id"],
                "question": question,
                "ground_truth": row["ground_truth"],
                "llm_answer": answer,
            }
        )
        print(f"[{row['id']}] 완료: {question[:40]}...")

    baseline_df = pd.DataFrame(rows)
    baseline_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ LLM baseline 저장: {out_path} ({len(baseline_df)}건)")
    return baseline_df


def load_eval_dataset(eval_path: Path | str | None = None) -> pd.DataFrame:
    """RAG eval_dataset.csv 로드."""
    path = Path(eval_path) if eval_path else DEFAULT_EVAL_DATASET
    if not path.exists():
        raise FileNotFoundError(
            f"RAG 평가 데이터셋을 찾을 수 없습니다: {path}\n"
            "먼저 python -m src.make_eval_dataset 를 실행하세요."
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def build_comparison(
    baseline_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """LLM baseline과 RAG eval_dataset을 id 기준 병합."""
    rag_cols = ["id", "rag_answer", "retrieved_contexts", "source_hint"]
    missing = set(rag_cols) - set(eval_df.columns)
    if missing:
        raise ValueError(f"eval_dataset.csv에 필수 컬럼이 없습니다: {missing}")

    merged = baseline_df.merge(
        eval_df[rag_cols],
        on="id",
        how="inner",
        validate="one_to_one",
    )

    merged["llm_has_source"] = merged["llm_answer"].apply(has_source_expression)
    merged["rag_has_source"] = merged["rag_answer"].apply(has_source_expression)
    merged["llm_answer_length"] = merged["llm_answer"].astype(str).str.len()
    merged["rag_answer_length"] = merged["rag_answer"].astype(str).str.len()

    comparison_df = merged[
        [
            "id",
            "question",
            "ground_truth",
            "llm_answer",
            "rag_answer",
            "retrieved_contexts",
            "source_hint",
            "llm_has_source",
            "rag_has_source",
            "llm_answer_length",
            "rag_answer_length",
        ]
    ]

    out_path = Path(output_path) if output_path else DEFAULT_COMPARISON
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ LLM vs RAG 비교 CSV 저장: {out_path}")
    return comparison_df


def write_summary(
    comparison_df: pd.DataFrame,
    summary_path: Path | str | None = None,
) -> None:
    """비교 실험 요약 텍스트 저장."""
    out_path = Path(summary_path) if summary_path else DEFAULT_SUMMARY
    n = len(comparison_df)

    llm_source_count = int(comparison_df["llm_has_source"].sum())
    rag_source_count = int(comparison_df["rag_has_source"].sum())
    avg_llm_len = comparison_df["llm_answer_length"].mean()
    avg_rag_len = comparison_df["rag_answer_length"].mean()

    ragas_path = PROJECT_ROOT / "outputs" / "ragas_results.csv"
    ragas_section = ""
    if ragas_path.exists():
        ragas_df = pd.read_csv(ragas_path, encoding="utf-8-sig")
        ragas_lines = ["", "[RAGAS 평가 결과 (기존 run_ragas_eval.py)]"]
        for _, row in ragas_df.iterrows():
            ragas_lines.append(f"  - {row['metric']}: {row['score']:.3f}")
        ragas_section = "\n".join(ragas_lines)

    lines = [
        "=" * 60,
        "순수 LLM vs RAG 답변 비교 실험 요약",
        "=" * 60,
        f"총 질문 수: {n}",
        "",
        "[출처/근거 표현 포함 비율]",
        f"  - LLM baseline: {llm_source_count}/{n} ({llm_source_count / n:.1%})",
        f"  - RAG:         {rag_source_count}/{n} ({rag_source_count / n:.1%})",
        "",
        "[평균 답변 길이 (문자 수)]",
        f"  - LLM baseline: {avg_llm_len:.1f}",
        f"  - RAG:         {avg_rag_len:.1f}",
        "",
        "[실험 해석 가이드]",
        "  - RAG는 검색된 공식 문서 context 기반으로 답변합니다.",
        "  - 순수 LLM은 Vector DB/Retriever 없이 GPT만 사용합니다.",
        "  - rag_has_source > llm_has_source 이면 RAG가 근거 제시에 유리함을 시사합니다.",
        "  - 정량적 품질 평가는 outputs/ragas_results.csv (RAGAS)를 참고하세요.",
        ragas_section,
        "",
        "[출력 파일]",
        f"  - {DEFAULT_LLM_BASELINE.name}",
        f"  - {DEFAULT_COMPARISON.name}",
        f"  - {DEFAULT_SUMMARY.name}",
        "=" * 60,
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 요약 저장: {out_path}")


def run_llm_vs_rag_experiment(
    ground_truth_path: Path | str | None = None,
    eval_dataset_path: Path | str | None = None,
    baseline_path: Path | str | None = None,
    comparison_path: Path | str | None = None,
    summary_path: Path | str | None = None,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> pd.DataFrame:
    """순수 LLM baseline 생성 → RAG와 병합 → 요약 저장."""
    baseline_df = build_llm_baseline(
        ground_truth_path=ground_truth_path,
        output_path=baseline_path,
        llm_model=llm_model,
    )
    eval_df = load_eval_dataset(eval_dataset_path)
    comparison_df = build_comparison(
        baseline_df=baseline_df,
        eval_df=eval_df,
        output_path=comparison_path,
    )
    write_summary(comparison_df, summary_path=summary_path)
    return comparison_df


def main() -> None:
    parser = argparse.ArgumentParser(description="순수 LLM vs RAG 비교 실험")
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="ground_truth.csv 경로",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        default=None,
        help="RAG eval_dataset.csv 경로 (기본: outputs/eval_dataset.csv)",
    )
    parser.add_argument(
        "--baseline-output",
        type=str,
        default=None,
        help="LLM baseline CSV 경로",
    )
    parser.add_argument(
        "--comparison-output",
        type=str,
        default=None,
        help="비교 CSV 경로",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="요약 TXT 경로",
    )
    args = parser.parse_args()

    run_llm_vs_rag_experiment(
        ground_truth_path=args.ground_truth,
        eval_dataset_path=args.eval_dataset,
        baseline_path=args.baseline_output,
        comparison_path=args.comparison_output,
        summary_path=args.summary_output,
    )


if __name__ == "__main__":
    main()
