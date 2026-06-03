"""
ground_truth.csv 기반 RAG 평가 데이터셋 생성 모듈.
(ragas_guide.py PART 3, 200~214줄 패턴 적용)

실행 방법:
    python -m src.make_eval_dataset
    python -m src.make_eval_dataset --top_k 2
    python -m src.make_eval_dataset --all-topk

환경변수:
    OPENAI_API_KEY  (.env 또는 시스템 환경변수)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from ragas import SingleTurnSample

from src.rag_chain import get_rag_answer

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "data" / "ground_truth.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "eval_dataset.csv"
TOPK_EXPERIMENT_VALUES = [2, 3, 5]


def eval_dataset_path_for_topk(top_k: int) -> Path:
    """top_k별 eval_dataset 출력 경로."""
    return PROJECT_ROOT / "outputs" / f"eval_dataset_k{top_k}.csv"


def load_ground_truth(csv_path: Path | str | None = None) -> pd.DataFrame:
    """ground_truth.csv를 로드한다."""
    path = Path(csv_path) if csv_path else DEFAULT_GROUND_TRUTH
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"id", "question", "ground_truth", "source_hint"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ground_truth.csv에 필수 컬럼이 없습니다: {missing}")
    if len(df) < 10:
        raise ValueError("ground_truth.csv에는 10개 이상의 질문이 필요합니다.")
    return df


def resolve_output_path(output_path: Path | str | None, top_k: int) -> Path:
    """출력 경로 결정: 명시 경로 > top_k 실험 경로 > 기본 경로."""
    if output_path:
        return Path(output_path)
    if top_k in TOPK_EXPERIMENT_VALUES:
        return eval_dataset_path_for_topk(top_k)
    return DEFAULT_OUTPUT


def build_eval_dataset(
    ground_truth_path: Path | str | None = None,
    output_path: Path | str | None = None,
    top_k: int = 4,
) -> pd.DataFrame:
    """
    각 질문에 대해 RAG 답변을 생성하고 eval_dataset.csv로 저장한다.
    """
    df = load_ground_truth(ground_truth_path)
    out_path = resolve_output_path(output_path, top_k)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n🔄 RAG 답변 생성 중... (top_k={top_k})")
    samples = []
    rows = []

    # ═══════════════════════════════════════════════════════════════
    # ★ ragas_guide.py 200~214줄 대응 구간
    #   - CSV 컬럼명이 다르면 아래 question / ground_truth 매핑만 수정하세요.
    #   - guide 원본: user_input → question, reference → ground_truth
    # ═══════════════════════════════════════════════════════════════
    for _, row in df.iterrows():
        question = row["question"]          # guide: row["user_input"]
        ground_truth = row["ground_truth"]  # guide: row["reference"]

        # RAG 시스템으로 답변 생성
        result = get_rag_answer(question, top_k=top_k)

        # RAGAS 평가용 샘플 생성
        sample = SingleTurnSample(
            user_input=question,
            response=result["answer"],
            retrieved_contexts=result["contexts"],
            reference=ground_truth,
        )
        samples.append(sample)

        rows.append(
            {
                "id": row["id"],
                "question": question,
                "ground_truth": ground_truth,
                "source_hint": row["source_hint"],
                "top_k": top_k,
                "rag_answer": result["answer"],
                "retrieved_contexts": json.dumps(
                    result["contexts"], ensure_ascii=False
                ),
            }
        )
        print(f"[{row['id']}] 완료: {question[:40]}...")

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 평가 데이터셋 저장: {out_path} ({len(eval_df)}건)")
    print(f"✅ RAGAS 샘플 준비 완료: {len(samples)}개")
    return eval_df


def build_all_topk_datasets(
    top_k_values: list[int] | None = None,
    ground_truth_path: Path | str | None = None,
) -> None:
    """top-k 실험용 eval_dataset_k*.csv 일괄 생성."""
    values = top_k_values or TOPK_EXPERIMENT_VALUES
    for k in values:
        print(f"\n{'=' * 50}")
        print(f"top_k={k} 데이터셋 생성")
        print(f"{'=' * 50}")
        build_eval_dataset(
            ground_truth_path=ground_truth_path,
            output_path=eval_dataset_path_for_topk(k),
            top_k=k,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 평가 데이터셋 생성")
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="Retriever 검색 chunk 수 (기본값: 4 → outputs/eval_dataset.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 CSV 경로 (미지정 시 top_k에 따라 자동 결정)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="ground_truth.csv 경로",
    )
    parser.add_argument(
        "--all-topk",
        action="store_true",
        help="top_k=2,3,5 실험용 데이터셋 일괄 생성",
    )
    args = parser.parse_args()

    if args.all_topk:
        build_all_topk_datasets(ground_truth_path=args.ground_truth)
    else:
        build_eval_dataset(
            ground_truth_path=args.ground_truth,
            output_path=args.output,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
