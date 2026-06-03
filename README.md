# workcheck-rag

공식 노동자료 기반 근로계약·근무관리 RAG 서비스의 **RAGAS 리포트 제출용 평가 파이프라인**입니다.

## 서비스 개요

| 항목 | 내용 |
|------|------|
| 주제 | 근무체크: 공식 노동자료 기반 근로계약·근무관리 RAG |
| 대상 | 아르바이트 근로자, 소규모 매장 점주, 근무표·시급 관리 담당자 |
| 근거 | 고용노동부 공식 문서 (PDF/TXT) |

## 폴더 구조

```
workcheck-rag/
├── data/
│   ├── official_docs/       # 공식 문서 (PDF/TXT)
│   ├── processed/
│   └── ground_truth.csv     # 평가용 정답 12문항
├── vectorstore/faiss_index/ # FAISS 벡터 DB
├── src/                     # 파이프라인 스크립트
├── outputs/                 # 평가 결과
├── app.py                   # Streamlit UI
└── requirements.txt
```

## 사전 준비

1. Python 3.10+ 권장
2. (권장) 가상환경 생성 후 의존성 설치

```bash
cd workcheck-rag
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> LangChain 1.x와 RAGAS 0.4.x 간 import 충돌을 피하기 위해 `requirements.txt`에 호환 버전 범위가 지정되어 있습니다.

3. 환경변수 설정

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key"

# Linux/macOS
export OPENAI_API_KEY="your-api-key"
```

4. 공식 문서 배치 (`data/official_docs/` 하위)

| 경로 | 문서 |
|------|------|
| `contract/2025_표준근로계약서_고용노동부.pdf` | 2025 개정 표준근로계약서 |
| `contract/근로계약서_작성방법_고용노동부.txt` | 근로계약서 작성방법 안내 |
| `minimum_wage/2026_최저임금_고용노동부.pdf` | 2026년 적용 최저임금 |
| `working_time/근로시간_휴게시간_FAQ_고용노동부.txt` | 근로시간·휴게시간 FAQ |

> PDF가 없을 경우 동일 내용의 TXT 샘플이 포함되어 있어 파이프라인 테스트가 가능합니다.

## 실행 순서

### 1. 문서 ingest 확인 (선택)

```bash
python -m src.ingest_docs
```

### 2. FAISS 벡터 DB 생성

```bash
python -m src.build_vector_db
```

### 3. RAG 단일 질의 테스트 (선택)

```bash
python -m src.rag_chain "2026년 적용 최저임금 시간급은 얼마인가?"
python -m src.rag_chain "2026년 적용 최저임금 시간급은 얼마인가?" --top_k 3
```

### 4. 평가 데이터셋 생성

```bash
python -m src.make_eval_dataset
```

→ `outputs/eval_dataset.csv` 저장

### 5. RAGAS 평가 실행

```bash
python -m src.run_ragas_eval
```

→ `outputs/ragas_results.csv`, `outputs/ragas_chart.png` 저장

### 6. Top-k 성능 비교 실험 (논문용)

동일 `ground_truth.csv` 12문항, `top_k ∈ {2, 3, 5}` 조건으로 RAGAS 성능을 비교합니다.

**데이터셋 생성 (top_k별 RAG 답변)**

```bash
python -m src.make_eval_dataset --top_k 2
python -m src.make_eval_dataset --top_k 3
python -m src.make_eval_dataset --top_k 5

# 또는 일괄 생성
python -m src.make_eval_dataset --all-topk
```

→ `outputs/eval_dataset_k2.csv`, `eval_dataset_k3.csv`, `eval_dataset_k5.csv`

**RAGAS 평가 (top_k별)**

```bash
python -m src.run_ragas_eval --input outputs/eval_dataset_k2.csv --output outputs/ragas_results_k2.csv --no-chart
python -m src.run_ragas_eval --input outputs/eval_dataset_k3.csv --output outputs/ragas_results_k3.csv --no-chart
python -m src.run_ragas_eval --input outputs/eval_dataset_k5.csv --output outputs/ragas_results_k5.csv --no-chart
```

**Top-k 비교 리포트 생성**

```bash
python -m src.run_ragas_eval --compare-topk
```

→ `outputs/topk_comparison.csv`, `outputs/topk_comparison_chart.png`

**평가 + 비교 일괄 실행** (eval_dataset_k*.csv가 이미 있을 때)

```bash
python -m src.run_ragas_eval --eval-all-topk
```

### 7. 순수 LLM vs RAG 비교 실험 (논문용)

동일 `ground_truth.csv` 12문항에 대해 **Retriever 없는 순수 LLM**과 **RAG** 답변을 비교합니다.
(RAGAS 평가는 기존 `run_ragas_eval.py` 결과를 사용)

**사전 조건:** `outputs/eval_dataset.csv` 생성 완료

```bash
python -m src.make_eval_dataset
python -m src.make_llm_baseline
```

**출력 파일**

| 파일 | 내용 |
|------|------|
| `outputs/llm_baseline_answers.csv` | 순수 LLM 답변 (id, question, ground_truth, llm_answer) |
| `outputs/llm_vs_rag_comparison.csv` | LLM vs RAG 병합 + 비교 지표 |
| `outputs/llm_vs_rag_summary.txt` | 출처 표현 비율·답변 길이 요약 |

**비교 지표 (llm_vs_rag_comparison.csv)**
- `llm_has_source` / `rag_has_source`: 출처·근거 표현 포함 여부
- `llm_answer_length` / `rag_answer_length`: 답변 길이(문자 수)

### 8. Streamlit 앱 실행

```bash
streamlit run app.py
```

## RAGAS 평가 지표

| 지표 | 설명 |
|------|------|
| Faithfulness | 답변이 검색 context에 충실한지 |
| Answer Relevancy | 답변이 질문과 관련 있는지 |
| Context Precision | 검색 context의 정밀도 |
| Context Recall | 정답 대비 context 회수율 |

## RAG 답변 정책

- 검색된 **공식 문서 context만** 근거로 답변
- 문서에 없는 내용 → `"문서에서 확인할 수 없습니다"`
- 법률·노무 **자문처럼 단정하지 않음**
- `"공식 자료 기준으로 확인해야 할 항목"` 표현 사용
- 가능하면 **근거 문서명** 포함

## 파일별 역할

| 파일 | 역할 |
|------|------|
| `src/ingest_docs.py` | PDF/TXT 로드, chunk 분할 |
| `src/build_vector_db.py` | OpenAI Embedding + FAISS 저장 |
| `src/rag_chain.py` | Retriever + GPT RAG 답변 |
| `src/make_eval_dataset.py` | ground_truth → eval_dataset.csv |
| `src/make_llm_baseline.py` | 순수 LLM baseline + RAG 비교 CSV |
| `src/run_ragas_eval.py` | RAGAS 평가 + 차트 생성 |

## 출력 파일

| 파일 | 내용 |
|------|------|
| `outputs/eval_dataset.csv` | 질문, RAG 답변, context, 정답 |
| `outputs/ragas_results.csv` | 4개 지표 점수 |
| `outputs/ragas_chart.png` | 지표 막대그래프 |
| `outputs/eval_dataset_k{2,3,5}.csv` | Top-k 실험용 평가 데이터셋 |
| `outputs/ragas_results_k{2,3,5}.csv` | Top-k별 RAGAS 결과 |
| `outputs/topk_comparison.csv` | Top-k별 4지표 + 평균 비교 |
| `outputs/topk_comparison_chart.png` | Top-k별 성능 비교 막대그래프 |
| `outputs/llm_baseline_answers.csv` | 순수 LLM baseline 답변 |
| `outputs/llm_vs_rag_comparison.csv` | LLM vs RAG 비교 (출처·길이 지표 포함) |
| `outputs/llm_vs_rag_summary.txt` | LLM vs RAG 비교 요약 |

## 주의사항

- API 키는 코드에 포함하지 않으며 `OPENAI_API_KEY` 환경변수만 사용합니다.
- RAGAS는 `EvaluationDataset` + `SingleTurnSample` 방식을 우선 사용하며, 실패 시 `datasets.Dataset.from_dict` fallback을 지원합니다.
- 본 서비스는 참고용 정보 제공 목적이며, 법률·노무 자문을 대체하지 않습니다.
