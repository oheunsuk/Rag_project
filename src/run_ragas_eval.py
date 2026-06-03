"""
RAGAS 평가 실행 및 결과 시각화 모듈.
(ragas_guide.py PART 3 평가 방식 적용)

실행 방법:
    python -m src.run_ragas_eval
    python -m src.run_ragas_eval --input outputs/eval_dataset_k2.csv --output outputs/ragas_results_k2.csv
    python -m src.run_ragas_eval --compare-topk

환경변수:
    OPENAI_API_KEY  (.env 또는 시스템 환경변수)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_DATASET = PROJECT_ROOT / "outputs" / "eval_dataset.csv"
DEFAULT_RESULTS = PROJECT_ROOT / "outputs" / "ragas_results.csv"
DEFAULT_CHART = PROJECT_ROOT / "outputs" / "ragas_chart.png"
TOPK_COMPARISON_CSV = PROJECT_ROOT / "outputs" / "topk_comparison.csv"
TOPK_COMPARISON_CHART = PROJECT_ROOT / "outputs" / "topk_comparison_chart.png"
TOPK_EXPERIMENT_VALUES = [2, 3, 5]

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}


def ragas_results_path_for_topk(top_k: int) -> Path:
    return PROJECT_ROOT / "outputs" / f"ragas_results_k{top_k}.csv"


def load_eval_dataset(csv_path: Path | str | None = None) -> pd.DataFrame:
    """eval_dataset.csv를 로드한다."""
    path = Path(csv_path) if csv_path else DEFAULT_EVAL_DATASET
    if not path.exists():
        raise FileNotFoundError(
            f"평가 데이터셋을 찾을 수 없습니다: {path}\n"
            "먼저 python -m src.make_eval_dataset 를 실행하세요."
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def _parse_contexts(raw: str) -> List[str]:
    if pd.isna(raw) or raw == "":
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(c) for c in parsed]
    except json.JSONDecodeError:
        pass
    return [str(raw)]


def _build_samples_from_csv(df: pd.DataFrame) -> List[Any]:
    """
    eval_dataset.csv → SingleTurnSample 리스트.
    (make_eval_dataset.py 200~214줄과 동일 구조)
    """
    from ragas import SingleTurnSample

    samples = []
    for _, row in df.iterrows():
        question = row["question"]
        ground_truth = row["ground_truth"]
        sample = SingleTurnSample(
            user_input=str(question),
            response=str(row["rag_answer"]),
            retrieved_contexts=_parse_contexts(row["retrieved_contexts"]),
            reference=str(ground_truth),
        )
        samples.append(sample)
    return samples


def _get_evaluator_llm():
    """ragas_guide.py 방식: LangchainLLMWrapper + ChatOpenAI."""
    from ragas.llms import LangchainLLMWrapper

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    return LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
    )


def _get_metric_classes(evaluator_llm):
    """ragas_guide.py Faithfulness/AnswerRelevancy 등 클래스 방식."""
    try:
        from ragas.metrics import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )

        return [
            Faithfulness(llm=evaluator_llm),
            AnswerRelevancy(llm=evaluator_llm),
            ContextPrecision(llm=evaluator_llm),
            ContextRecall(llm=evaluator_llm),
        ]
    except ImportError:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        return [faithfulness, answer_relevancy, context_precision, context_recall]


def _run_ragas_with_evaluation_dataset(samples: List[Any]) -> Any:
    from ragas import EvaluationDataset, evaluate

    evaluator_llm = _get_evaluator_llm()
    metrics = _get_metric_classes(evaluator_llm)

    eval_dataset = EvaluationDataset(samples=samples)
    return evaluate(dataset=eval_dataset, metrics=metrics)


def _run_ragas_with_hf_dataset(df: pd.DataFrame) -> Any:
    """EvaluationDataset 방식 실패 시 datasets.Dataset fallback."""
    from datasets import Dataset
    from ragas import evaluate

    evaluator_llm = _get_evaluator_llm()
    metrics = _get_metric_classes(evaluator_llm)

    hf_dataset = Dataset.from_dict(
        {
            "question": df["question"].astype(str).tolist(),
            "answer": df["rag_answer"].astype(str).tolist(),
            "contexts": [_parse_contexts(c) for c in df["retrieved_contexts"]],
            "ground_truth": df["ground_truth"].astype(str).tolist(),
        }
    )
    return evaluate(dataset=hf_dataset, metrics=metrics)


def _extract_metric_scores(result: Any) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for name in METRIC_NAMES:
        value = None
        if hasattr(result, name):
            value = getattr(result, name)
        elif isinstance(result, dict) and name in result:
            value = result[name]
        elif hasattr(result, "to_pandas"):
            pdf = result.to_pandas()
            if name in pdf.columns:
                value = pdf[name].mean()
        if value is not None:
            scores[name] = float(value)
    return scores


def _scores_to_results_df(scores: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": METRIC_LABELS.get(k, k), "score": v} for k, v in scores.items()]
    )


def run_ragas_evaluation(
    eval_dataset_path: Path | str | None = None,
    results_path: Path | str | None = None,
    chart_path: Path | str | None = None,
    save_chart: bool = True,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    """RAGAS 평가를 실행하고 결과 CSV·차트를 저장한다."""
    df = load_eval_dataset(eval_dataset_path)
    out_results = Path(results_path) if results_path else DEFAULT_RESULTS
    out_chart = Path(chart_path) if chart_path else DEFAULT_CHART
    out_results.parent.mkdir(parents=True, exist_ok=True)

    samples = _build_samples_from_csv(df)

    try:
        print("RAGAS 평가 실행 (EvaluationDataset + LangchainLLMWrapper)...")
        result = _run_ragas_with_evaluation_dataset(samples)
    except Exception as exc:
        print(f"EvaluationDataset 방식 실패, fallback 사용: {exc}")
        print("RAGAS 평가 실행 (datasets.Dataset 방식)...")
        result = _run_ragas_with_hf_dataset(df)

    scores = _extract_metric_scores(result)
    if not scores:
        raise RuntimeError("RAGAS 평가 결과를 추출하지 못했습니다.")

    if hasattr(result, "to_pandas"):
        detail_df = result.to_pandas()
        detail_path = out_results.with_name(
            out_results.stem + "_detail" + out_results.suffix
        )
        detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
        print(f"세부 결과 저장: {detail_path}")

    results_df = _scores_to_results_df(scores)
    results_df.to_csv(out_results, index=False, encoding="utf-8-sig")

    if save_chart:
        _save_single_chart(scores, out_chart)

    print("\n" + "=" * 50)
    print("📊 RAGAS 평가 결과")
    print("=" * 50)
    for name in METRIC_NAMES:
        if name in scores:
            print(f"  {METRIC_LABELS[name]:22s}: {scores[name]:.3f}")
    avg = sum(scores.values()) / len(scores)
    print("=" * 50)
    print(f"  평균                    : {avg:.3f}")
    print(f"\n✅ 결과 저장: {out_results}")
    if save_chart:
        print(f"✅ 차트 저장: {out_chart}")

    return results_df, scores


def _load_ragas_scores(results_path: Path) -> Dict[str, float]:
    """ragas_results*.csv에서 metric→score dict 로드."""
    df = pd.read_csv(results_path, encoding="utf-8-sig")
    label_to_key = {v: k for k, v in METRIC_LABELS.items()}
    scores: Dict[str, float] = {}
    for _, row in df.iterrows():
        metric_label = str(row["metric"])
        key = label_to_key.get(metric_label, metric_label.lower().replace(" ", "_"))
        scores[key] = float(row["score"])
    return scores


def build_topk_comparison(
    top_k_values: list[int] | None = None,
    comparison_csv: Path | str | None = None,
    comparison_chart: Path | str | None = None,
) -> pd.DataFrame:
    """
    top-k별 RAGAS 결과를 비교 CSV·막대그래프로 저장한다.
    """
    values = top_k_values or TOPK_EXPERIMENT_VALUES
    out_csv = Path(comparison_csv) if comparison_csv else TOPK_COMPARISON_CSV
    out_chart = Path(comparison_chart) if comparison_chart else TOPK_COMPARISON_CHART
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for k in values:
        results_path = ragas_results_path_for_topk(k)
        if not results_path.exists():
            raise FileNotFoundError(
                f"RAGAS 결과 파일이 없습니다: {results_path}\n"
                f"먼저 python -m src.run_ragas_eval --input outputs/eval_dataset_k{k}.csv "
                f"--output outputs/ragas_results_k{k}.csv 를 실행하세요."
            )
        scores = _load_ragas_scores(results_path)
        row = {"top_k": k}
        for name in METRIC_NAMES:
            row[name] = scores.get(name)
        row["average"] = sum(scores.values()) / len(scores) if scores else None
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    _save_topk_comparison_chart(comparison_df, out_chart)

    print(f"\n✅ Top-k 비교 CSV 저장: {out_csv}")
    print(f"✅ Top-k 비교 차트 저장: {out_chart}")
    print(comparison_df.to_string(index=False))
    return comparison_df


def run_all_topk_evaluations(
    top_k_values: list[int] | None = None,
) -> pd.DataFrame:
    """top-k별 eval → RAGAS 평가 → 비교 리포트 일괄 실행."""
    values = top_k_values or TOPK_EXPERIMENT_VALUES
    for k in values:
        eval_path = PROJECT_ROOT / "outputs" / f"eval_dataset_k{k}.csv"
        results_path = ragas_results_path_for_topk(k)
        print(f"\n{'=' * 50}")
        print(f"top_k={k} RAGAS 평가")
        print(f"{'=' * 50}")
        run_ragas_evaluation(
            eval_dataset_path=eval_path,
            results_path=results_path,
            chart_path=PROJECT_ROOT / "outputs" / f"ragas_chart_k{k}.png",
            save_chart=False,
        )
    return build_topk_comparison(top_k_values=values)


def _save_single_chart(scores: Dict[str, float], chart_path: Path) -> None:
    labels = [METRIC_LABELS.get(k, k) for k in scores.keys()]
    values = list(scores.values())
    avg_score = sum(values) / len(values)

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    )
    ax.axhline(
        y=avg_score, color="gray", linestyle="--", label=f"평균: {avg_score:.3f}"
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Workcheck RAG - RAGAS 평가 결과")
    ax.legend()

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def _save_topk_comparison_chart(comparison_df: pd.DataFrame, chart_path: Path) -> None:
    """top-k별 4개 지표 평균 점수 그룹 막대그래프."""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    top_k_labels = [f"k={k}" for k in comparison_df["top_k"]]
    x = range(len(top_k_labels))
    width = 0.18
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(METRIC_NAMES):
        offset = (i - 1.5) * width
        values = comparison_df[metric].tolist()
        bars = ax.bar(
            [pos + offset for pos in x],
            values,
            width,
            label=METRIC_LABELS[metric],
            color=colors[i],
        )
        for bar, val in zip(bars, values):
            if pd.notna(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(list(x))
    ax.set_xticklabels(top_k_labels)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Score")
    ax.set_title("Top-k별 RAGAS 성능 비교")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 평가 실행")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="입력 eval_dataset CSV (기본: outputs/eval_dataset.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 ragas_results CSV (기본: outputs/ragas_results.csv)",
    )
    parser.add_argument(
        "--chart",
        type=str,
        default=None,
        help="출력 차트 PNG (기본: outputs/ragas_chart.png)",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="단일 평가 시 차트 저장 생략",
    )
    parser.add_argument(
        "--compare-topk",
        action="store_true",
        help="top_k=2,3,5 결과를 비교 CSV·차트 생성",
    )
    parser.add_argument(
        "--eval-all-topk",
        action="store_true",
        help="top_k=2,3,5 RAGAS 평가 + 비교 리포트 일괄 실행",
    )
    args = parser.parse_args()

    if args.eval_all_topk:
        run_all_topk_evaluations()
    elif args.compare_topk:
        build_topk_comparison()
    else:
        run_ragas_evaluation(
            eval_dataset_path=args.input,
            results_path=args.output,
            chart_path=args.chart,
            save_chart=not args.no_chart,
        )


if __name__ == "__main__":
    main()
