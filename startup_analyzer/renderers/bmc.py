import datetime
import html
from typing import Any, Dict, List

import streamlit as st

from startup_analyzer.services.bmc import ensure_bmc_shape
from startup_analyzer.utils.text import normalize_text_list


def _bmc_items(bmc: Dict[str, Any], key: str, limit: int = 6) -> List[str]:
    return normalize_text_list((bmc or {}).get(key, []), limit=limit)


def _bmc_cell_html(bmc: Dict[str, Any], title: str, key: str) -> str:
    items = _bmc_items(bmc, key)
    bullets = "<br>".join(f"- {html.escape(x)}" for x in items) if items else "- 확인 불가"
    return (
        "<div style='font-size:20px;font-weight:700;color:#111827;margin-bottom:8px;'>"
        f"{html.escape(title)}</div>"
        "<div style='font-size:14px;line-height:1.5;color:#334155;'>"
        f"{bullets}</div>"
    )


def render_bmc(data: Dict[str, Any]):
    bmc = ensure_bmc_shape(data).get("business_model_canvas", {})
    table_html = f"""
    <table style="border-collapse:collapse;width:100%;table-layout:fixed;background:#fff;">
      <tr>
        <td rowspan="2" style="border:2px solid #222;padding:12px;vertical-align:top;">{_bmc_cell_html(bmc, "핵심 파트너", "key_partnerships")}</td>
        <td style="border:2px solid #222;padding:12px;vertical-align:top;">{_bmc_cell_html(bmc, "핵심 활동", "key_activities")}</td>
        <td rowspan="2" style="border:2px solid #222;padding:12px;vertical-align:top;">{_bmc_cell_html(bmc, "가치 제안", "value_propositions")}</td>
        <td style="border:2px solid #222;padding:12px;vertical-align:top;">{_bmc_cell_html(bmc, "고객 관계", "customer_relationships")}</td>
        <td rowspan="2" style="border:2px solid #222;padding:12px;vertical-align:top;">{_bmc_cell_html(bmc, "고객 세그먼트", "customer_segments")}</td>
      </tr>
      <tr>
        <td style="border:2px solid #222;padding:12px;vertical-align:top;">{_bmc_cell_html(bmc, "핵심 자원", "key_resources")}</td>
        <td style="border:2px solid #222;padding:12px;vertical-align:top;">{_bmc_cell_html(bmc, "채널", "channels")}</td>
      </tr>
      <tr>
        <td colspan="3" style="border:2px solid #222;padding:12px;vertical-align:top;">{_bmc_cell_html(bmc, "비용 구조", "cost_structure")}</td>
        <td colspan="2" style="border:2px solid #222;padding:12px;vertical-align:top;">{_bmc_cell_html(bmc, "수익 흐름", "revenue_streams")}</td>
      </tr>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def build_bmc_markdown(data: Dict[str, Any]) -> str:
    bmc = ensure_bmc_shape(data).get("business_model_canvas", {})
    rows = [
        ("고객 세그먼트", "customer_segments"),
        ("가치 제안", "value_propositions"),
        ("채널", "channels"),
        ("고객 관계", "customer_relationships"),
        ("수익 흐름", "revenue_streams"),
        ("핵심 자원", "key_resources"),
        ("핵심 활동", "key_activities"),
        ("핵심 파트너", "key_partnerships"),
        ("비용 구조", "cost_structure"),
    ]
    lines = ["# Business Model Canvas", ""]
    for title, key in rows:
        lines.append(f"## {title}")
        values = _bmc_items(bmc, key)
        if values:
            lines.extend([f"- {value}" for value in values])
        else:
            lines.append("- 확인 불가")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_overview_report_markdown(
    company_name: str,
    ceo_name: str,
    profile: Dict[str, Any],
    keywords: List[str],
    bmc_data: Dict[str, Any],
) -> str:
    data = ensure_bmc_shape(bmc_data, company_name=company_name)
    summary = data.get("strategic_summary", {})
    lines = [
        f"# {company_name} 기업개요 리포트",
        "",
        f"- 대표자: {ceo_name}",
        f"- 생성일: {datetime.date.today()}",
        "",
        "## 기업 분석",
        "",
        "### 문제 정의",
        profile.get("problem_definition", ""),
        "",
        "### 솔루션 및 제공 가치",
        profile.get("solution_value_prop", ""),
        "",
        "### 비즈니스 모델",
        profile.get("revenue_model_type", ""),
        "",
        "### 핵심 기능",
    ]
    for item in normalize_text_list(profile.get("product_core_features", []), limit=8):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "### 핵심 기술 · 경쟁력",
            profile.get("core_tech_moat", ""),
            "",
            "### 대표자 비전",
            profile.get("ceo_vision_summary", ""),
            "",
            "### 조직 · 운영 방식",
            profile.get("org_culture_biz_focus", ""),
            "",
            "### 최근 뉴스 요약",
            profile.get("recent_news_summary", ""),
            "",
            "### 산업 키워드",
            ", ".join(keywords),
            "",
            "## 비즈니스모델 캔버스 요약",
            data.get("bmc_summary", ""),
            "",
            "## 전략 요약",
            "",
            f"- 문제: {summary.get('problem', '')}",
            f"- 기존 대안의 한계: {summary.get('status_quo', '')}",
            f"- 우리 해법: {summary.get('our_solution', '')}",
            "",
        ]
    )
    return "\n".join(lines).strip() + "\n"
