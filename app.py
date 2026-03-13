import os
import re
import json
import html
import datetime
from typing import Optional, List, Dict, Any

import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types


MODEL_NAME = "gemini-2.0-flash"


def get_gemini_api_key() -> Optional[str]:
    try:
        key = st.secrets.get("GEMINI_API_KEY", None) or st.secrets.get("GOOGLE_API_KEY", None)
        if key:
            return str(key).strip()
    except Exception:
        pass

    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return str(key).strip()

    st.error(
        "Gemini API Key가 설정되지 않았습니다.\n\n"
        "로컬 실행:\n"
        "  - 터미널에서 `export GEMINI_API_KEY=\"YOUR_KEY\"` 설정 후 실행\n\n"
        "Streamlit Cloud 배포:\n"
        "  - App settings → Secrets 에 아래처럼 추가\n"
        "    GEMINI_API_KEY=\"YOUR_KEY\"\n"
        "    GOOGLE_API_KEY=\"YOUR_KEY\" (선택)"
    )
    return None


def _escape_inner_quotes_heuristic(text: str) -> str:
    out = []
    in_str = False
    esc = False
    i = 0
    n = len(text)

    def next_non_space(idx: int) -> str:
        j = idx
        while j < n and text[j].isspace():
            j += 1
        return text[j] if j < n else ""

    while i < n:
        ch = text[i]

        if esc:
            out.append(ch)
            esc = False
            i += 1
            continue

        if ch == "\\":
            out.append(ch)
            esc = True
            i += 1
            continue

        if ch == '"':
            if not in_str:
                in_str = True
                out.append(ch)
            else:
                nxt = next_non_space(i + 1)
                if nxt in [",", "}", "]"]:
                    in_str = False
                    out.append(ch)
                else:
                    out.append('\\"')
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty response")

    cleaned = text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON block not found")

    raw = cleaned[start:end + 1]
    try:
        return json.loads(raw)
    except Exception:
        repaired = _escape_inner_quotes_heuristic(raw)
        return json.loads(repaired)


def repair_json_with_model(client: genai.Client, raw_text: str, schema_hint: str = "") -> str:
    fix_prompt = (
        "아래 출력은 JSON 형식이 깨져 있습니다. 내용을 최대한 동일하게 유지하되,\n"
        "반드시 표준 JSON으로만 수정해서 JSON만 출력하세요.\n\n"
        "[규칙]\n"
        "- 코드펜스, 설명 문장, 주석 금지\n"
        "- 문자열 내부 큰따옴표는 JSON 문법에 맞게 처리\n"
        "- 키 이름과 구조는 유지\n"
        "- JSON ONLY\n\n"
    )
    if schema_hint:
        fix_prompt += f"[스키마 참고]\n{schema_hint}\n\n"
    fix_prompt += f"[원본]\n{raw_text}\n"

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=fix_prompt,
        config=types.GenerateContentConfig(response_mime_type="text/plain"),
    )
    return (response.text or "").strip()


def normalize_text_list(values: Any, limit: int = 6) -> List[str]:
    if isinstance(values, list):
        raw_items = values
    elif values:
        raw_items = [values]
    else:
        raw_items = []

    output = []
    seen = set()
    for item in raw_items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def ensure_bmc_shape(data: Dict[str, Any]) -> Dict[str, Any]:
    output = dict(data or {})
    output["bm_type"] = str(output.get("bm_type", "")).strip()
    output["bmc_summary"] = str(output.get("bmc_summary", "")).strip()

    summary = dict(output.get("strategic_summary", {}) or {})
    output["strategic_summary"] = {
        "problem": str(summary.get("problem", "")).strip(),
        "status_quo": str(summary.get("status_quo", "")).strip(),
        "our_solution": str(summary.get("our_solution", "")).strip(),
    }

    output["top_layer"] = normalize_text_list(output.get("top_layer", []), limit=3)
    output["left_actors"] = normalize_text_list(output.get("left_actors", []), limit=4)
    output["right_actors"] = normalize_text_list(output.get("right_actors", []), limit=4)
    output["middle_layer"] = str(output.get("middle_layer", "")).strip()

    def normalize_flow_items(items: Any) -> List[Dict[str, str]]:
        normalized = []
        for item in items or []:
            if isinstance(item, dict):
                normalized.append(
                    {
                        "from": str(item.get("from", "")).strip(),
                        "to": str(item.get("to", "")).strip(),
                        "label": str(item.get("label", "")).strip(),
                    }
                )
            else:
                text = str(item or "").strip()
                if text:
                    normalized.append({"from": "", "to": "", "label": text})
            if len(normalized) >= 8:
                break
        return normalized

    output["money_flows"] = normalize_flow_items(output.get("money_flows", []))
    output["information_flows"] = normalize_flow_items(output.get("information_flows", []))
    output["service_flows"] = normalize_flow_items(output.get("service_flows", []))

    bmc = dict(output.get("business_model_canvas", {}) or {})
    output["business_model_canvas"] = {
        "customer_segments": normalize_text_list(bmc.get("customer_segments", [])),
        "value_propositions": normalize_text_list(bmc.get("value_propositions", [])),
        "channels": normalize_text_list(bmc.get("channels", [])),
        "customer_relationships": normalize_text_list(bmc.get("customer_relationships", [])),
        "revenue_streams": normalize_text_list(bmc.get("revenue_streams", [])),
        "key_resources": normalize_text_list(bmc.get("key_resources", [])),
        "key_activities": normalize_text_list(bmc.get("key_activities", [])),
        "key_partnerships": normalize_text_list(bmc.get("key_partnerships", [])),
        "cost_structure": normalize_text_list(bmc.get("cost_structure", [])),
    }
    return output


def render_step(step: int):
    if step == 1:
        html_block = """
<div style="display:flex;gap:12px;margin-bottom:24px;">
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 1 · 정보 수집</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#F1F5F9;color:#475569;">
STEP 2 · JSON 분석</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#F1F5F9;color:#475569;">
STEP 3 · 결과 생성</div>
</div>
"""
    elif step == 2:
        html_block = """
<div style="display:flex;gap:12px;margin-bottom:24px;">
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 1 · 정보 수집</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 2 · JSON 분석</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#F1F5F9;color:#475569;">
STEP 3 · 결과 생성</div>
</div>
"""
    else:
        html_block = """
<div style="display:flex;gap:12px;margin-bottom:24px;">
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 1 · 정보 수집</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 2 · JSON 분석</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 3 · 결과 생성</div>
</div>
"""
    st.write(html_block, unsafe_allow_html=True)


def tile(title: str, body: str):
    safe = str(body or "").replace("\n", "<br>")
    st.write(
        f"""
<div style="background:#E2E8F0;padding:10px 14px;border-radius:6px 6px 0 0;
border:1px solid #CBD5E1;font-weight:600;">{title}</div>""",
        unsafe_allow_html=True,
    )
    st.write(
        f"""
<div style="background:white;padding:16px;border-radius:0 0 6px 6px;
border:1px solid #CBD5E1;border-top:none;font-size:14px;line-height:1.6;">
{safe}</div>""",
        unsafe_allow_html=True,
    )
    st.write("<div style='height:14px;'></div>", unsafe_allow_html=True)


def extract_keywords(profile: Dict[str, Any]) -> List[str]:
    kws = [k for k in profile.get("industry_keywords", []) if "확인 불가" not in str(k)]
    if kws:
        return kws

    features = profile.get("product_core_features", [])
    text = " ".join([str(x) for x in features]) if isinstance(features, list) else str(features)
    auto = [token for token in text.lower().split() if len(token) > 3]
    return list(dict.fromkeys(auto))[:5] if auto else ["technology"]


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
    data = ensure_bmc_shape(bmc_data)
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


def safe_filename(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", str(text or "").strip())
    return value.strip("_") or "report"


def _wrap_svg_text(text: str, max_chars: int = 18, max_lines: int = 2) -> List[str]:
    value = str(text or "").strip()
    if not value:
        return [""]

    lines = []
    cur = ""
    for ch in value:
        if len(cur + ch) <= max_chars:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
            if len(lines) >= max_lines - 1:
                break
    if cur:
        lines.append(cur)

    lines = lines[:max_lines]
    consumed = "".join(lines)
    if len(consumed) < len(value) and lines:
        lines[-1] = lines[-1][: max(0, max_chars - 3)] + "..."
    return lines


def _svg_multiline_text(
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    font_size: int = 24,
    fill: str = "#253240",
    weight: str = "700",
) -> str:
    lines = _wrap_svg_text(text, max_chars=14, max_lines=2)
    line_h = font_size + 7
    total_h = line_h * len(lines) - 7
    start_y = y + (h - total_h) / 2 + font_size * 0.82
    tspans = []
    for idx, line in enumerate(lines):
        tspans.append(f'<tspan x="{x + w / 2}" y="{start_y + idx * line_h}">{html.escape(line)}</tspan>')
    return (
        '<text text-anchor="middle" '
        'font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" '
        f'font-size="{font_size}" font-weight="{weight}" fill="{fill}">'
        + "".join(tspans)
        + "</text>"
    )


def _svg_rect_node(node_id: str, x: float, y: float, w: float, h: float, label: str, icon: str) -> str:
    return (
        f'<g id="{node_id}">'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="20" ry="20" fill="#dfeaf7" stroke="#60a5dd" stroke-width="3"/>'
        f'<text x="{x + 24}" y="{y + 34}" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" '
        f'font-size="28" fill="#536273">{html.escape(icon)}</text>'
        f"{_svg_multiline_text(x + 18, y + 14, w - 36, h - 18, label)}"
        "</g>"
    )


def _edge_anchor(box: tuple, target_box: tuple) -> tuple:
    x, y, w, h = box
    tx, ty, tw, th = target_box
    cx, cy = x + w / 2, y + h / 2
    tcx, tcy = tx + tw / 2, ty + th / 2
    dx, dy = tcx - cx, tcy - cy
    if abs(dx) > abs(dy):
        return (x + w, cy) if dx > 0 else (x, cy)
    return (cx, y + h) if dy > 0 else (cx, y)


def _svg_flow_path(start_box: tuple, end_box: tuple, color: str, marker_id: str, label: str, symbol: str) -> str:
    sx, sy = _edge_anchor(start_box, end_box)
    ex, ey = _edge_anchor(end_box, start_box)
    mx = (sx + ex) / 2
    my = (sy + ey) / 2
    dx, dy = ex - sx, ey - sy

    if abs(dx) > abs(dy):
        cy = my - 40 if dy >= 0 else my + 40
        path = f"M {sx} {sy} C {mx} {sy}, {mx} {cy}, {ex} {ey}"
        lx, ly = mx, cy - 10
    else:
        cx = mx - 56 if dx >= 0 else mx + 56
        path = f"M {sx} {sy} C {sx} {my}, {cx} {my}, {ex} {ey}"
        lx, ly = cx, my - 12

    text_label = f"{symbol} {label}".strip()
    return (
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="4" marker-end="url(#{marker_id})"/>'
        f'<text x="{lx}" y="{ly}" text-anchor="middle" '
        'font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" '
        f'font-size="22" font-weight="600" fill="{color}">{html.escape(text_label)}</text>'
    )


def _first_non_empty(values: Any, fallback: str = "") -> str:
    if isinstance(values, list):
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
        return fallback
    text = str(values or "").strip()
    return text or fallback


def _resolve_node_key(name: str, node_labels: Dict[str, str]) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    for key, label in node_labels.items():
        if text == label:
            return key
    for key, label in node_labels.items():
        if text in label or label in text:
            return key
    return ""


def build_editable_svg(data: Dict[str, Any], company_name: str) -> bytes:
    source = ensure_bmc_shape(data)
    width, height = 1600, 900
    node_boxes = {
        "top": (620, 90, 260, 100),
        "left": (70, 335, 280, 112),
        "center": (635, 325, 270, 124),
        "right": (1250, 335, 280, 112),
        "bottom": (620, 610, 280, 100),
    }

    top_label = _first_non_empty(source.get("top_layer", []), _first_non_empty(source["business_model_canvas"].get("customer_segments", []), "핵심 고객"))
    left_label = _first_non_empty(source.get("left_actors", []), _first_non_empty(source["business_model_canvas"].get("key_partnerships", []), "핵심 파트너"))
    center_label = _first_non_empty(source.get("middle_layer", ""), "핵심 플랫폼")
    right_label = _first_non_empty(source.get("right_actors", []), _first_non_empty(source["business_model_canvas"].get("channels", []), "전달 채널"))
    bottom_label = company_name

    node_labels = {
        "top": top_label,
        "left": left_label,
        "center": center_label,
        "right": right_label,
        "bottom": bottom_label,
    }

    fallback_pairs = [
        ("top", "center"),
        ("center", "top"),
        ("left", "center"),
        ("center", "left"),
        ("center", "right"),
        ("right", "center"),
        ("bottom", "center"),
        ("center", "bottom"),
    ]

    flow_fragments = []
    for items, color, marker_id, symbol in [
        (source.get("money_flows", []), "#4b9e58", "arrow-green", "$"),
        (source.get("information_flows", []), "#3b82db", "arrow-blue", "□"),
        (source.get("service_flows", []), "#e58b2c", "arrow-orange", "○"),
    ]:
        for idx, item in enumerate(items[: len(fallback_pairs)]):
            start_key = _resolve_node_key(item.get("from", ""), node_labels)
            end_key = _resolve_node_key(item.get("to", ""), node_labels)
            if not start_key or not end_key:
                start_key, end_key = fallback_pairs[idx]
            if start_key not in node_boxes or end_key not in node_boxes:
                continue
            flow_fragments.append(
                _svg_flow_path(node_boxes[start_key], node_boxes[end_key], color, marker_id, item.get("label", ""), symbol)
            )

    legend_frag = (
        '<text x="500" y="830" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" '
        'font-size="18" font-weight="700" fill="#4b5c74">범례:</text>'
        '<text x="560" y="830" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" '
        'font-size="18" font-weight="700" fill="#4b9e58">$ = 돈 흐름</text>'
        '<text x="720" y="830" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" '
        'font-size="18" font-weight="700" fill="#3b82db">□ = 정보 흐름</text>'
        '<text x="920" y="830" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" '
        'font-size="18" font-weight="700" fill="#e58b2c">○ = 서비스/자산 흐름</text>'
    )

    title = f"{html.escape(company_name)} BM 다이어그램"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrow-green" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
    <path d="M 0 0 L 12 6 L 0 12 z" fill="#4b9e58"/>
  </marker>
  <marker id="arrow-blue" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
    <path d="M 0 0 L 12 6 L 0 12 z" fill="#3b82db"/>
  </marker>
  <marker id="arrow-orange" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">
    <path d="M 0 0 L 12 6 L 0 12 z" fill="#e58b2c"/>
  </marker>
</defs>
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
<text x="40" y="46" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" font-size="28" font-weight="700" fill="#283241">{title}</text>
{''.join(flow_fragments)}
{_svg_rect_node("node-top", *node_boxes["top"], top_label, "◌")}
{_svg_rect_node("node-left", *node_boxes["left"], left_label, "◌")}
{_svg_rect_node("node-center", *node_boxes["center"], center_label, "◎")}
{_svg_rect_node("node-right", *node_boxes["right"], right_label, "◌")}
{_svg_rect_node("node-bottom", *node_boxes["bottom"], bottom_label, "◌")}
{legend_frag}
</svg>"""
    return svg.encode("utf-8")


def render_svg_preview(svg_bytes: bytes, height: int = 920):
    components.html(svg_bytes.decode("utf-8"), height=height, scrolling=True)


def build_bmc_and_diagram_data(
    client: genai.Client,
    company_name: str,
    ceo_name: str,
    facts: str,
    profile: Dict[str, Any],
    keywords: List[str],
) -> Dict[str, Any]:
    schema_hint = """
{
  "bm_type": "",
  "bmc_summary": "",
  "strategic_summary": {
    "problem": "",
    "status_quo": "",
    "our_solution": ""
  },
  "top_layer": [],
  "middle_layer": "",
  "left_actors": [],
  "right_actors": [],
  "money_flows": [{"from": "", "to": "", "label": ""}],
  "information_flows": [{"from": "", "to": "", "label": ""}],
  "service_flows": [{"from": "", "to": "", "label": ""}],
  "business_model_canvas": {
    "customer_segments": [],
    "value_propositions": [],
    "channels": [],
    "customer_relationships": [],
    "revenue_streams": [],
    "key_resources": [],
    "key_activities": [],
    "key_partnerships": [],
    "cost_structure": []
  }
}
""".strip()

    prompt = f"""
당신은 스타트업 사업모델 분석가이다.

아래 자료를 바탕으로 {company_name}의 비즈니스모델 캔버스와 BM 다이어그램 구조를 재구성하라.

[입력 정보]
- 기업명: {company_name}
- 대표자명: {ceo_name}
- 산업 키워드: {", ".join(keywords)}

[사실 정보]
{facts}

[기존 기업 분석 JSON]
{json.dumps(profile, ensure_ascii=False, indent=2)}

[핵심 목표]
1. 정식 Business Model Canvas 9블록을 완성한다.
2. SVG 생태계형 BM 다이어그램에 필요한 노드와 흐름을 만든다.
3. 기업개요 리포트에 넣을 BMC 요약 문단을 작성한다.

[작성 규칙]
- JSON ONLY
- 모든 일반 라벨은 한국어로 작성
- business_model_canvas의 각 항목은 짧고 구체적인 한국어 명사구로 작성
- 추정이 필요하면 보수적으로 작성
- 과도하게 일반적인 표현 금지
- top_layer, left_actors, right_actors는 다이어그램 노드용 짧은 명사구 사용
- middle_layer는 중앙 플랫폼 또는 핵심 엔진을 가장 잘 설명하는 짧은 명사구 사용
- money_flows, information_flows, service_flows의 각 항목은 반드시 from, to, label을 포함
- flow label은 2~6자 내외의 짧은 한국어 표현
- bmc_summary는 2~3문장 분량의 한국어 문단
- strategic_summary의 각 필드는 한 문장으로 작성

[다이어그램 설계 원칙]
- 상단 노드는 핵심 고객 또는 수요 주체
- 좌측 노드는 공급/제휴/인프라 측 핵심 주체
- 중앙 노드는 플랫폼/솔루션/핵심 운영 엔진
- 우측 노드는 유통/채널/도입 조직/수요처 중 핵심 주체
- 하단 노드는 반드시 기업명 자체로 표현되며 JSON에 넣지 않아도 됨
- 흐름은 돈, 정보, 서비스/자산의 의미가 드러나야 함

[출력 스키마]
{schema_hint}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="text/plain"),
    )
    raw_text = (response.text or "").strip()

    try:
        return ensure_bmc_shape(extract_json(raw_text))
    except Exception:
        repaired = repair_json_with_model(client, raw_text, schema_hint=schema_hint)
        return ensure_bmc_shape(extract_json(repaired))


st.set_page_config(layout="wide", page_title="혁신의숲 Startup Analyzer & Report")

st.write(
    """
<div style="width:100%;padding:26px 0;text-align:center;">
    <div style="margin-bottom:4px;">
        <span style="color:#00D2A8;font-size:32px;font-weight:700;">
            혁신의숲 Startup Analyzer & Report
        </span>
    </div>
    <div>
        <span style="color:#64748B;font-size:14px;font-weight:500;">
            Powered by Mark & Company
        </span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 설정")
    st.caption("API Key는 화면 입력이 아니라 Secrets/환경변수에서 읽습니다.")
    with st.expander("로컬/배포 설정 방법", expanded=False):
        st.code(
            '로컬:\n'
            '  export GEMINI_API_KEY="YOUR_KEY"\n'
            '  python -m streamlit run app.py\n\n'
            'Streamlit Cloud Secrets:\n'
            '  GEMINI_API_KEY="YOUR_KEY"\n'
            '  GOOGLE_API_KEY="YOUR_KEY"  # 선택\n',
            language="bash",
        )
    st.markdown("---")

st.markdown("## 기업 정보 입력")

col1, col2 = st.columns(2)
with col1:
    company_name = st.text_input("기업명", placeholder="예: 마크앤컴퍼니")
with col2:
    ceo_name = st.text_input("대표자명", placeholder="예: 홍경표")

raw_text = st.text_area(
    "보조 텍스트 (뉴스/메모 등)",
    height=130,
    placeholder="기업과 관련된 기사나 참고 텍스트를 입력하세요. (선택)",
)

run = st.button("분석 실행", type="primary")

if run:
    if not company_name.strip():
        st.error("기업명을 입력해주세요.")
        st.stop()
    if not ceo_name.strip():
        st.error("대표자명을 입력해주세요.")
        st.stop()

    api_key = get_gemini_api_key()
    if not api_key:
        st.stop()

    client = genai.Client(api_key=api_key)
    google_tool = types.Tool(google_search=types.GoogleSearch())

    render_step(1)
    gather_prompt = f"""
회사명 {company_name}, 대표자 {ceo_name}에 대한 사실 기반 정보를 Google 검색으로 수집하라.

[규칙]
- 검증된 사실만 작성
- 대표자 인터뷰/발언이 있으면 반드시 포함
- 추측, 요약, 해석 금지
- JSON 금지
- 텍스트만 출력
"""

    gather_resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=gather_prompt,
        config=types.GenerateContentConfig(
            tools=[google_tool],
            response_mime_type="text/plain",
        ),
    )
    facts = (gather_resp.text or "").strip()
    if raw_text.strip():
        facts = f"{facts}\n\n[사용자 보조 텍스트]\n{raw_text.strip()}"

    render_step(2)
    profile_schema_hint = """
{
  "problem_definition": "",
  "solution_value_prop": "",
  "revenue_model_type": "",
  "product_core_features": [],
  "core_tech_moat": "",
  "ceo_vision_summary": "",
  "org_culture_biz_focus": "",
  "recent_news_summary": "",
  "industry_keywords": []
}
""".strip()

    json_prompt = f"""
아래는 {company_name}에 관한 사실 기반 정보이다:
{facts}

아래 기준에 따라 기업 분석 JSON만 생성하라.

[기업 분석 지침]
- 객관적, 분석적 전문가 문체
- 특수문자("*","**","~") 금지
- 각 항목 최소 120자 이상
- 기업명 기반 뻔한 설명 금지
- 대표자 비전은 공신력 있는 출처 기반
- 조직문화는 채용사이트 표현 금지
- 불확실한 정보는 확인 불가
- 추정은 (추정됨) 또는 (예상됨) 명시
- 광고성 표현 금지
- 문자열 값 내부 큰따옴표 사용 금지
- JSON ONLY

[출력 스키마]
{profile_schema_hint}
"""

    json_resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=json_prompt,
        config=types.GenerateContentConfig(response_mime_type="text/plain"),
    )
    raw_json_text = (json_resp.text or "").strip()

    try:
        profile = extract_json(raw_json_text)
    except Exception:
        st.warning("기업 분석 JSON 파싱 오류가 발생해 자동 정정을 시도합니다.")
        fixed_text = repair_json_with_model(client, raw_json_text, schema_hint=profile_schema_hint)
        try:
            profile = extract_json(fixed_text)
        except Exception:
            st.error("기업 분석 JSON 자동 정정에도 실패했습니다.")
            st.code(raw_json_text)
            st.stop()

    keywords = extract_keywords(profile)
    try:
        bmc_data = build_bmc_and_diagram_data(client, company_name, ceo_name, facts, profile, keywords)
    except Exception as exc:
        st.error(f"BMC 및 BM 다이어그램 데이터 생성에 실패했습니다: {exc}")
        st.stop()
    diagram_svg = build_editable_svg(bmc_data, company_name)
    overview_report_md = build_overview_report_markdown(company_name, ceo_name, profile, keywords, bmc_data)
    bmc_md = build_bmc_markdown(bmc_data)

    render_step(3)
    st.markdown("## 기업 분석 결과")
    tile("문제 정의", profile.get("problem_definition", ""))
    tile("솔루션 및 제공 가치", profile.get("solution_value_prop", ""))
    tile("비즈니스 모델", profile.get("revenue_model_type", ""))
    tile("핵심 기능", "<br>".join(normalize_text_list(profile.get("product_core_features", []), limit=8)))
    tile("핵심 기술 · 경쟁력", profile.get("core_tech_moat", ""))
    tile("대표자 비전", profile.get("ceo_vision_summary", ""))
    tile("조직 · 운영 방식", profile.get("org_culture_biz_focus", ""))
    tile("최근 뉴스 요약", profile.get("recent_news_summary", ""))
    tile("산업 키워드", ", ".join(keywords))

    st.markdown("## Business Model Canvas")
    if bmc_data.get("bm_type"):
        tile("BM 유형", bmc_data.get("bm_type", ""))
    if bmc_data.get("bmc_summary"):
        tile("BMC 요약", bmc_data.get("bmc_summary", ""))
    render_bmc(bmc_data)

    st.markdown("## BM 다이어그램")
    render_svg_preview(diagram_svg)

    st.markdown("## 다운로드")
    safe_name = safe_filename(company_name)
    today = datetime.date.today()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button(
            "기업개요 리포트 다운로드",
            data=overview_report_md,
            file_name=f"Company_Overview_{safe_name}_{today}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_b:
        st.download_button(
            "BMC 다운로드",
            data=bmc_md,
            file_name=f"BMC_{safe_name}_{today}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_c:
        st.download_button(
            "BM 다이어그램 다운로드",
            data=diagram_svg,
            file_name=f"BM_Diagram_{safe_name}_{today}.svg",
            mime="image/svg+xml",
            use_container_width=True,
        )
