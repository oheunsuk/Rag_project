"""
계약서·스케줄 점검 데모 로직 (세션 내 전용, 공식 RAG DB 미사용).

공식 기준은 get_rag_answer()로 별도 검색한다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from src.pdf_utils import extract_text_from_pdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "samples"
FALLBACK_CONTRACT = SAMPLES_DIR / "fallback" / "sample_contract_text.txt"
FALLBACK_SCHEDULE = SAMPLES_DIR / "fallback" / "sample_schedule_text.txt"
SAMPLE_CONTRACT_PDF = SAMPLES_DIR / "contracts" / "sample_contract.pdf"
SAMPLE_SCHEDULE_PDF = SAMPLES_DIR / "schedules" / "sample_weekly_schedule.pdf"

OFFICIAL_MIN_WAGE = 10320
MIN_PDF_TEXT_LEN = 50

REQUIRED_FIELDS = [
    ("근로개시일", ["근로개시일", "근로계약기간", "계약기간"]),
    ("근무장소", ["근무장소", "근 무 장 소"]),
    ("업무내용", ["업무내용", "업무의 내용", "업무의 내용(직종)"]),
    ("소정근로시간", ["소정근로시간", "근로시간"]),
    ("휴게시간", ["휴게시간", "휴게 시간"]),
    ("근무일", ["근무일", "근로일"]),
    ("주휴일", ["주휴일"]),
    ("임금", ["임금", "시간급", "시급", "월급", "일급"]),
    ("임금지급일", ["임금지급일", "지급일", "급여지급일"]),
    ("지급방법", ["지급방법", "계좌에 입금", "예금통장에 입금", "예금통장 입금"]),
    ("연차유급휴가", ["연차유급휴가", "연차 유급휴가"]),
    (
        "근로계약서 교부",
        ["근로계약서 교부", "근로자에게 교부", "사본을 교부", "계약서 교부"],
    ),
]

RAG_EVIDENCE_QUESTIONS = [
    "2026년 적용 최저임금 시간급은 얼마인가요?",
    "근로기준법상 근로시간에 따른 법정 최소 휴게시간 기준은 무엇인가요?",
    "표준근로계약서에서 확인해야 할 주요 항목은 무엇인가요?",
]

SHOW_DEV_SOURCE_INFO = False

__all__ = [
    "FALLBACK_CONTRACT",
    "FALLBACK_SCHEDULE",
    "RAG_EVIDENCE_QUESTIONS",
    "SAMPLE_CONTRACT_PDF",
    "SAMPLE_SCHEDULE_PDF",
    "SHOW_DEV_SOURCE_INFO",
    "check_break_time",
    "check_minimum_wage",
    "check_required_fields",
    "check_schedule",
    "format_field_status",
    "get_source_display_message",
    "resolve_document_text",
    "resolve_example_document",
    "resolve_uploaded_pdf",
]

_TIME_RANGE_PATTERN = re.compile(
    r"(\d{1,2})\s*(?:시|:)\s*(\d{1,2})?\s*분?"
    r"\s*(?:부터|~|-)\s*"
    r"(\d{1,2})\s*(?:시|:)\s*(\d{1,2})?\s*분?"
    r"(?:까지)?"
)


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _normalize_nospace(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def _split_lines(text: str) -> List[str]:
    return [
        normalize_line(line)
        for line in text.splitlines()
        if normalize_line(line)
    ]


def _read_fallback(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def resolve_document_text(
    uploaded_file,
    sample_pdf_path: Path,
    fallback_path: Path,
) -> Tuple[str, str, bool]:
    """
    Returns:
        (text, source, pdf_extract_failed)
        source: uploaded_pdf | sample_pdf | fallback
    """
    pdf_extract_failed = False
    fallback_text = _read_fallback(fallback_path)

    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file.getvalue())
        if pdf_text and len(pdf_text.strip()) >= MIN_PDF_TEXT_LEN:
            return pdf_text, "uploaded_pdf", False
        pdf_extract_failed = True
        if fallback_text:
            return fallback_text, "fallback", True
        return "", "fallback", True

    if sample_pdf_path.exists():
        pdf_text = extract_text_from_pdf(sample_pdf_path)
        if pdf_text and len(pdf_text.strip()) >= MIN_PDF_TEXT_LEN:
            return pdf_text, "sample_pdf", False

    if fallback_text:
        return fallback_text, "fallback", False

    return "", "fallback", False


def parse_time_range(line: str) -> Tuple[int, int] | None:
    line = normalize_line(line)
    match = _TIME_RANGE_PATTERN.search(line)
    if not match:
        return None

    start_hour = int(match.group(1))
    start_minute = int(match.group(2) or 0)
    end_hour = int(match.group(3))
    end_minute = int(match.group(4) or 0)

    start = start_hour * 60 + start_minute
    end = end_hour * 60 + end_minute

    if end < start:
        end += 24 * 60

    return start, end


def _minutes_to_hhmm(total_minutes: int) -> str:
    hour = (total_minutes // 60) % 24
    minute = total_minutes % 60
    return f"{hour:02d}:{minute:02d}"


def _range_to_display(time_range: Tuple[int, int]) -> str:
    start, end = time_range
    return f"{_minutes_to_hhmm(start)}~{_minutes_to_hhmm(end)}"


def _parse_work_and_break_ranges(lines: List[str]) -> Tuple[Tuple[int, int] | None, Tuple[int, int] | None]:
    work_range = None
    break_range = None

    for line in lines:
        compact = line.replace(" ", "")

        if work_range is None and (
            "근로시간:" in compact or "소정근로시간:" in compact
        ):
            parsed = parse_time_range(line)
            if parsed:
                work_range = parsed

        if break_range is None and (
            "휴게시간:" in compact or compact.startswith("휴게:")
        ):
            parsed = parse_time_range(line)
            if parsed:
                break_range = parsed

    return work_range, break_range


def required_break_minutes(actual_work_minutes: int) -> int:
    if actual_work_minutes >= 8 * 60:
        return 60
    if actual_work_minutes >= 4 * 60:
        return 30
    return 0


def _format_duration_minutes(minutes: int) -> str:
    if minutes <= 0:
        return "0분"
    if minutes % 60 == 0:
        return f"{minutes // 60}시간"
    return f"{minutes}분"


def _format_work_hours_display(minutes: int) -> str:
    if minutes <= 0:
        return "0시간"
    hours = minutes / 60
    if hours == int(hours):
        return f"{int(hours)}시간"
    return f"{hours:g}시간"


def extract_workplace(lines: List[str]) -> str | None:
    for index, line in enumerate(lines):
        compact = line.replace(" ", "")

        is_label_line = compact.startswith("근무장소")
        is_section_header = bool(re.match(r"^\d+\.", line)) and "근무장소" in compact

        if not (is_label_line or is_section_header):
            continue

        if is_label_line and ":" in line:
            value = line.split(":", 1)[1].strip()
            if value and "공란" not in value.replace(" ", ""):
                return value

        if index + 1 < len(lines):
            next_line = lines[index + 1].strip()
            next_compact = next_line.replace(" ", "")

            if re.match(r"^\d+\.", next_line):
                return None

            if (
                next_compact.startswith("업무의내용")
                or next_compact.startswith("업무내용")
            ):
                return None

            if "공란" in next_compact:
                return None

            if next_line:
                return next_line

        return None

    return None


def _keyword_found(text: str, keywords: List[str]) -> bool:
    compact = normalize_text(text)
    nospace = _normalize_nospace(text)
    for kw in keywords:
        kw_nospace = re.sub(r"\s+", "", kw)
        if (
            kw in text
            or kw in compact
            or kw_nospace in nospace
        ):
            return True
    return False


def _parse_hourly_wage(text: str) -> int | None:
    compact = normalize_text(text)
    patterns = [
        r"시간급\s*[:：]?\s*([0-9,]+)\s*원",
        r"시급\s*[:：]?\s*([0-9,]+)\s*원",
        r"([0-9,]+)\s*원\s*/?\s*시간",
    ]
    for pat in patterns:
        m = re.search(pat, compact)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def _check_workplace_status(text: str) -> str:
    lines = _split_lines(text)
    has_label = any("근무장소" in line.replace(" ", "") for line in lines)
    if not has_label:
        return "missing"

    value = extract_workplace(lines)
    if value:
        return "ok"
    return "empty"


def check_minimum_wage(contract_text: str) -> Dict[str, str]:
    wage = _parse_hourly_wage(contract_text)
    if wage is None:
        return {
            "contract_wage": "확인 불가",
            "official_standard": f"2026년 적용 최저임금 시간급 {OFFICIAL_MIN_WAGE:,}원",
            "result": "계약서에서 시급을 찾지 못했습니다.",
        }
    meets = wage >= OFFICIAL_MIN_WAGE
    return {
        "contract_wage": f"{wage:,}원",
        "official_standard": f"2026년 적용 최저임금 시간급 {OFFICIAL_MIN_WAGE:,}원",
        "result": "기준 충족" if meets else "기준 미달 — 공식 자료 기준 확인 필요",
    }


def check_break_time(contract_text: str) -> Dict[str, str]:
    lines = _split_lines(contract_text)
    work_range, break_range = _parse_work_and_break_ranges(lines)

    work_display = _range_to_display(work_range) if work_range else "확인 불가"
    break_display = _range_to_display(break_range) if break_range else "확인 불가"

    if work_range is None and break_range is None:
        return {
            "work_hours": "확인 불가",
            "break_time": "확인 불가",
            "total_hours": "확인 불가",
            "break_duration": "확인 불가",
            "actual_work_hours": "확인 불가",
            "required_break": "확인 불가",
            "result": "자동 판정 불가",
            "result_detail": "근무·휴게시간을 계약서에서 찾지 못했습니다.",
        }

    total_span_minutes = None
    break_minutes = None
    actual_work_minutes = None

    if work_range:
        work_start, work_end = work_range
        total_span_minutes = work_end - work_start

    if break_range:
        break_start, break_end = break_range
        break_minutes = break_end - break_start

    if total_span_minutes is not None and break_minutes is not None:
        actual_work_minutes = total_span_minutes - break_minutes

    if (
        work_range is None
        or break_range is None
        or actual_work_minutes is None
    ):
        return {
            "work_hours": work_display,
            "break_time": break_display,
            "total_hours": (
                _format_work_hours_display(total_span_minutes)
                if total_span_minutes is not None
                else "확인 불가"
            ),
            "break_duration": (
                _format_duration_minutes(break_minutes)
                if break_minutes is not None
                else "확인 불가"
            ),
            "actual_work_hours": "확인 불가",
            "required_break": "확인 불가",
            "result": "자동 판정 불가",
            "result_detail": "근무시간 또는 휴게시간 정보가 불완전합니다.",
        }

    required = required_break_minutes(actual_work_minutes)
    meets = break_minutes >= required
    actual_hours = actual_work_minutes / 60

    if required == 0:
        result_detail = (
            f"실근로시간이 {actual_hours:g}시간이므로 법정 최소 휴게시간 의무가 없습니다."
        )
        if break_minutes > 0:
            result_detail += (
                f" 계약서에는 {_format_duration_minutes(break_minutes)}의 "
                "휴게시간이 기재되어 있습니다."
            )
    else:
        result_detail = (
            f"실근로시간이 {actual_hours:g}시간이므로 "
            f"법정 최소 휴게시간은 {required}분입니다. "
            f"계약서에는 {_format_duration_minutes(break_minutes)}의 "
            f"휴게시간이 기재되어 있어 "
            f"{'기준을 충족합니다.' if meets else '기준을 충족하지 못합니다.'}"
        )

    return {
        "work_hours": work_display,
        "break_time": break_display,
        "total_hours": _format_work_hours_display(total_span_minutes),
        "break_duration": _format_duration_minutes(break_minutes),
        "actual_work_hours": _format_work_hours_display(actual_work_minutes),
        "required_break": _format_duration_minutes(required),
        "result": "기준 충족" if meets else "기준 미달",
        "result_detail": result_detail,
    }


def check_required_fields(contract_text: str) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = []

    for label, keywords in REQUIRED_FIELDS:
        if label == "근무장소":
            results.append((label, _check_workplace_status(contract_text)))
            continue

        if _keyword_found(contract_text, keywords):
            results.append((label, "ok"))
        else:
            results.append((label, "missing"))

    return results


def format_field_status(label: str, status: str) -> str:
    if status == "ok":
        return f"✅ {label} 있음"
    if status == "empty":
        return f"❌ {label} 미기재"
    return f"❌ {label} 없음"


def resolve_uploaded_pdf(uploaded_file) -> Tuple[str, str, bool]:
    """업로드 PDF만 처리. 실패 시 fallback 사용하지 않음."""
    if uploaded_file is None:
        return "", "none", False

    pdf_text = extract_text_from_pdf(uploaded_file.getvalue())
    if pdf_text and len(pdf_text.strip()) >= MIN_PDF_TEXT_LEN:
        return pdf_text, "uploaded_pdf", False

    return "", "upload_failed", True


def resolve_example_document(
    sample_pdf_path: Path,
    fallback_path: Path,
) -> Tuple[str, str, bool]:
    """예시 문서(샘플 PDF 또는 fallback 텍스트)를 사용한다."""
    if sample_pdf_path.exists():
        pdf_text = extract_text_from_pdf(sample_pdf_path)
        if pdf_text and len(pdf_text.strip()) >= MIN_PDF_TEXT_LEN:
            return pdf_text, "example", False

    fallback_text = _read_fallback(fallback_path)
    if fallback_text:
        return fallback_text, "example", False

    return "", "none", False


def get_source_display_message(source: str) -> str:
    if source == "uploaded_pdf":
        return "업로드된 PDF에서 추출한 텍스트를 기준으로 점검했습니다."
    if source == "example":
        return "예시 문서를 기준으로 점검했습니다."
    if source == "upload_failed":
        return "업로드한 PDF에서 텍스트를 추출하지 못했습니다."
    if source == "none":
        return "문서가 선택되지 않았습니다."
    return ""


def check_schedule(schedule_text: str) -> Dict[str, str]:
    has_data = len(schedule_text.strip()) > 20
    return {
        "status": "근무표가 입력되었습니다." if has_data else "근무표 데이터 없음",
        "detail": "근무표에서 요일별·시간대별 근무 배치를 확인했습니다.",
        "note": "현재는 근무 배치 확인 기능을 제공하며, 세부 근로시간 분석은 지원 범위를 확대할 예정입니다.",
    }
