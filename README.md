# workcheck-rag

**근무체크 RAG** — 공식 노동자료 기반 근로계약·근무조건 RAG 서비스

## 프로젝트 개요

고용노동부 공식 노동자료를 FAISS Vector DB에 저장하고, 사용자 질문에 대해 **순수 LLM**과 **RAG** 답변을 동시에 비교해 보여줍니다. RAGAS 4지표로 정량 평가를 수행했으며, Top-k 비교 실험과 LLM vs RAG 비교 실험을 포함합니다.

## 서비스 대상

- 아르바이트 근로자
- 소규모 사업주
- 근무표·시급 관리 담당자

## 주요 기능

1. **순수 LLM vs RAG 비교** — 같은 질문에 2열 나란히 답변
2. **공식 문서 RAG** — FAISS + OpenAI Embedding 검색
3. **근거 확인** — 검색된 문서 chunk·출처 expander
4. **RAGAS 평가** — Faithfulness, Answer Relevancy, Context Precision, Context Recall
5. **Top-k 실험** — k=2, 3, 5 성능 비교
6. **대화 기록** — session_state 기반 이전 질문 저장
7. **계약서·스케줄 PDF 점검 데모** — 업로드 문서를 공식 RAG 기준과 비교 점검

## 순수 LLM vs RAG 비교

| 구분 | 순수 LLM | RAG |
|------|----------|-----|
| 문서 검색 | 없음 | FAISS 공식 문서 |
| 입력 | 질문만 | 질문 + 검색 chunk |
| 근거 | 일반 GPT 지식 | 문서명·chunk 인용 |
| 최신 수치 | 단정 지양 | 공식 고시 기반 |

→ **RAG가 공식 근거 확인에 유리**함을 UI에서 직접 비교

## 계약서·스케줄 PDF 점검 데모

- 사용자가 **근로계약서 PDF**와 **스케줄표 PDF**를 업로드하면 공식 노동자료 RAG 기준과 비교해 점검 결과를 보여줍니다.
- **업로드 PDF는 공식 RAG DB(FAISS)에 저장하지 않습니다.** 세션 내에서만 텍스트를 추출해 점검 대상으로 사용합니다.
- **공식 RAG 검색**에는 `data/docs/`의 고용노동부 공식 자료 3개만 사용합니다.
- PDF 텍스트 추출은 **`pypdf` 기반**이며, 스캔 이미지 PDF나 표 구조가 복잡한 PDF는 추출이 제한될 수 있습니다.
- 추출 실패 시 **`samples/fallback/`** 내장 샘플 텍스트를 사용해 데모가 중단되지 않습니다.
- **OCR 기반 PDF 분석**은 향후 확장 기능입니다.

## 사용 문서

| 파일 | 내용 |
|------|------|
| `2025_개정_표준근로계약서.txt` | 2025년 개정 표준근로계약서 |
| `2026_최저임금_보도자료.txt` | 2026년 적용 최저임금 고시 |
| `근로시간_휴게시간_FAQ.txt` | 근로시간·휴게시간 FAQ |

## 기술 스택

| 구분 | 기술 |
|------|------|
| UI | Streamlit |
| RAG | LangChain |
| Vector DB | FAISS |
| Embedding | OpenAI text-embedding-3-small |
| LLM | GPT-4o-mini |
| 평가 | RAGAS |

## 파일 구조

```
workcheck-rag/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── docs/
│   └── demo_script.md              # 발표 데모 멘트
├── src/
│   ├── __init__.py
│   ├── ingest_docs.py
│   ├── build_vector_db.py
│   ├── rag_chain.py
│   ├── llm_baseline.py
│   ├── make_eval_dataset.py
│   ├── run_ragas_eval.py
│   ├── pdf_utils.py              # pypdf 텍스트 추출
│   └── demo_check.py             # 계약서·스케줄 점검 로직
├── samples/
│   ├── contracts/                # 샘플 계약서 PDF (선택)
│   ├── schedules/                # 샘플 스케줄 PDF (선택)
│   └── fallback/                 # PDF 추출 실패 시 대체 텍스트
├── data/
│   ├── ground_truth.csv
│   └── docs/
│       ├── 2025_개정_표준근로계약서.txt
│       ├── 2026_최저임금_보도자료.txt
│       └── 근로시간_휴게시간_FAQ.txt
├── outputs/
│   ├── ragas_results.csv
│   ├── ragas_chart.png
│   ├── topk_comparison.csv
│   ├── topk_comparison_chart.png
│   ├── llm_vs_rag_comparison.csv
│   └── llm_vs_rag_summary.txt
└── vectorstore/
    └── faiss_index/
        ├── index.faiss
        └── index.pkl
```

### 폴더 역할

| 경로 | 역할 |
|------|------|
| `app.py` | Streamlit UI (챗봇·PDF 점검 데모 — 발표용 2탭) |
| `samples/` | 데모용 샘플 PDF·fallback 텍스트 (공식 RAG DB 미포함) |
| `src/` | 문서 ingest, 벡터 DB, RAG/LLM, RAGAS 파이프라인 |
| `data/docs/` | RAG 검색 대상 공식 문서 |
| `data/ground_truth.csv` | RAGAS 평가 질문 12개 |
| `outputs/` | RAGAS·Top-k·LLM vs RAG 결과 (발표용) |
| `vectorstore/faiss_index/` | 사전 구축 FAISS 인덱스 |

## 로컬 실행 방법

```powershell
cd workcheck-rag
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### 벡터 DB 재생성 (문서 변경 시)

```powershell
python -m src.build_vector_db
```

### RAGAS 평가 파이프라인

```powershell
python -m src.make_eval_dataset
python -m src.run_ragas_eval
```

## 환경변수 설정

API Key는 **코드에 직접 작성하지 않습니다.** 아래 순서로 읽습니다.

1. Streamlit secrets: `st.secrets["OPENAI_API_KEY"]`
2. 환경변수: `os.getenv("OPENAI_API_KEY")`
3. 로컬 `.env` (python-dotenv)

### 로컬 (.env)

프로젝트 루트에 `.env` 생성 (GitHub 업로드 금지):

```env
OPENAI_API_KEY=sk-your-key-here
```

### Streamlit Cloud (Secrets)

웹 설정 > **Secrets**에 등록:

```toml
OPENAI_API_KEY = "sk-..."
```

예시 파일: [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example)  
실제 `secrets.toml`은 업로드하지 않습니다.

## Streamlit Cloud 배포 방법

1. GitHub에 최종 코드가 올라간 repository 선택
2. [Streamlit Community Cloud](https://share.streamlit.io/)에서 **New app** 선택
3. Repository와 branch 선택
4. **Main file path**를 `app.py`로 설정
5. **Secrets**에 `OPENAI_API_KEY` 등록
6. **Deploy** 클릭
7. 생성된 배포 URL을 발표 PPT에 입력

> `vectorstore/faiss_index/`가 포함되어 있으면 별도 빌드 없이 바로 RAG 검색이 동작합니다.

## 배포 후 테스트

### 테스트 질문

1. `2026년 적용 최저임금 시간급은 얼마인가요?`
2. `8시간 근무하면 휴게시간은 얼마나 줘야 하나요?`
3. `근로계약서에 반드시 포함되어야 하는 항목은 무엇인가요?`

### 확인 항목

- [ ] 순수 LLM 답변이 출력되는가
- [ ] RAG 답변이 출력되는가
- [ ] 검색된 공식 문서 expander가 열리는가
- [ ] 사이드바에 챗봇·PDF 점검 데모 2개 메뉴만 표시되는가
- [ ] 이전 질문 기록이 저장되는가
- [ ] 계약서·스케줄 점검 데모 탭이 정상 동작하는가
- [ ] PDF 업로드 없이 fallback 텍스트로 점검이 진행되는가

## RAGAS 평가 결과

| 지표 | 점수 |
|------|------|
| Faithfulness | 0.884 |
| Answer Relevancy | 0.852 |
| Context Precision | 0.845 |
| Context Recall | 1.000 |
| **평균** | **0.895** |

## Top-k 비교 결과

| top_k | Faithfulness | Answer Relevancy | Context Precision | Context Recall | **평균** |
|-------|-------------|------------------|-------------------|----------------|----------|
| k=2 | 0.767 | 0.785 | 0.792 | 0.917 | **0.815** |
| k=3 | 0.830 | 0.786 | 0.812 | 1.000 | **0.857** |
| k=5 | 0.853 | 0.856 | 0.855 | 1.000 | **0.891** |

## 발표 데모

상세 멘트: [`docs/demo_script.md`](docs/demo_script.md)

### 발표 강조 포인트

- 순수 LLM과 RAG 답변을 같은 화면에서 비교
- RAG 답변은 공식 노동자료 chunk를 근거로 생성
- 검색된 문서와 출처를 사용자가 직접 확인 가능
- RAGAS 4지표로 정량 평가
- Ground Truth 12개 기준 평균 0.895
- Top-k 비교에서 k=5가 가장 높음
- 논문 제출 가점 항목 수행
- GitHub에는 실행·평가에 필요한 파일만 정리

## 링크

| 항목 | URL |
|------|-----|
| GitHub | TODO |
| 배포 (Streamlit Cloud) | TODO |

## 주의사항

- API Key는 코드·GitHub에 포함하지 않습니다.
- 본 서비스는 참고용이며, 법률·노무 자문을 대체하지 않습니다.
