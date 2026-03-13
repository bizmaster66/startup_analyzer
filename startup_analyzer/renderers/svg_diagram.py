import html
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import streamlit.components.v1 as components

from startup_analyzer.services.bmc import ensure_bmc_shape
from startup_analyzer.utils.text import clean_korean_label


CANVAS_WIDTH = 980
CANVAS_HEIGHT = 760
FRAME_X = 26
FRAME_Y = 20
FRAME_W = 928
FRAME_H = 706
HEADER_H = 54

NODE_BOXES = {
    "top": (420, 116, 140, 112),
    "left": (132, 304, 146, 124),
    "center": (422, 280, 136, 160),
    "right": (702, 304, 146, 124),
    "bottom": (414, 500, 152, 126),
}

NOTE_BOXES = {
    "lt": (62, 120, 136, 76),
    "rt": (784, 120, 136, 76),
    "lm": (54, 344, 142, 86),
    "rm": (784, 344, 142, 86),
}

def render_svg_preview(svg_bytes: bytes, height: int = 780):
    components.html(svg_bytes.decode("utf-8"), height=height, scrolling=True)


def build_editable_svg(data: Dict[str, Any], company_name: str) -> bytes:
    source = ensure_bmc_shape(data, company_name=company_name)
    node_labels = _build_node_labels(source, company_name)
    note_texts = _build_note_texts(source)
    subtitle = _build_subtitle(source)
    flows = _build_flow_specs(source, node_labels)

    flow_fragments = [_draw_flow(spec) for spec in flows]
    note_fragments = [
        _note_box(*NOTE_BOXES["lt"], note_texts["lt"]),
        _note_box(*NOTE_BOXES["rt"], note_texts["rt"]),
        _note_box(*NOTE_BOXES["lm"], note_texts["lm"]),
        _note_box(*NOTE_BOXES["rm"], note_texts["rm"]),
    ]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">
<defs>
  <marker id="arrow-black" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#111827"/>
  </marker>
</defs>
<rect x="0" y="0" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" fill="#ffffff"/>
{_frame()}
{_header(company_name, subtitle)}
{_section_lines()}
{''.join(note_fragments)}
{''.join(flow_fragments)}
{_entity_node("top", *NODE_BOXES["top"], node_labels["top"], role="top")}
{_entity_node("left", *NODE_BOXES["left"], node_labels["left"], role="left")}
{_entity_node("center", *NODE_BOXES["center"], node_labels["center"], role="center")}
{_entity_node("right", *NODE_BOXES["right"], node_labels["right"], role="right")}
{_company_node("bottom", *NODE_BOXES["bottom"], node_labels["bottom"])}
</svg>"""
    return svg.encode("utf-8")


def _frame() -> str:
    top_band = (
        f'<path d="M {FRAME_X+18} {FRAME_Y} H {FRAME_X+FRAME_W-18} '
        f'Q {FRAME_X+FRAME_W} {FRAME_Y} {FRAME_X+FRAME_W} {FRAME_Y+18} '
        f'V {FRAME_Y+HEADER_H} H {FRAME_X} V {FRAME_Y+18} '
        f'Q {FRAME_X} {FRAME_Y} {FRAME_X+18} {FRAME_Y} Z" fill="#47b7de"/>'
    )
    border = (
        f'<path d="M {FRAME_X+18} {FRAME_Y} H {FRAME_X+FRAME_W-18} '
        f'Q {FRAME_X+FRAME_W} {FRAME_Y} {FRAME_X+FRAME_W} {FRAME_Y+18} '
        f'V {FRAME_Y+FRAME_H-18} Q {FRAME_X+FRAME_W} {FRAME_Y+FRAME_H} {FRAME_X+FRAME_W-18} {FRAME_Y+FRAME_H} '
        f'H {FRAME_X+18} Q {FRAME_X} {FRAME_Y+FRAME_H} {FRAME_X} {FRAME_Y+FRAME_H-18} '
        f'V {FRAME_Y+18} Q {FRAME_X} {FRAME_Y} {FRAME_X+18} {FRAME_Y} Z" '
        f'fill="none" stroke="#47b7de" stroke-width="4"/>'
    )
    return top_band + border


def _header(company_name: str, subtitle: str) -> str:
    title = clean_korean_label(company_name, fallback=company_name)
    return (
        f'<text x="{FRAME_X+30}" y="{FRAME_Y+48}" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" '
        f'font-size="30" font-weight="900" fill="#111827">{html.escape(title)}</text>'
        f'<text x="{FRAME_X+30}" y="{FRAME_Y+76}" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" '
        f'font-size="14" font-weight="700" fill="#111827">{html.escape(subtitle)}</text>'
        f'<line x1="{FRAME_X+24}" y1="{FRAME_Y+94}" x2="{FRAME_X+FRAME_W-24}" y2="{FRAME_Y+94}" stroke="#47b7de" stroke-width="4"/>'
    )


def _section_lines() -> str:
    return (
        f'<line x1="{FRAME_X+22}" y1="254" x2="{FRAME_X+FRAME_W-22}" y2="254" stroke="#d7dbe0" stroke-width="1.5"/>'
        f'<line x1="{FRAME_X+22}" y1="470" x2="{FRAME_X+FRAME_W-22}" y2="470" stroke="#d7dbe0" stroke-width="1.5"/>'
    )


def _build_node_labels(source: Dict[str, Any], company_name: str) -> Dict[str, str]:
    return {
        "top": _short_label(source.get("top_layer", ["사용자"])[0], "사용자"),
        "left": _short_label(source.get("left_actors", ["핵심 자원"])[0], "핵심 자원"),
        "center": _short_label(source.get("middle_layer", "서비스"), "서비스"),
        "right": _short_label(source.get("right_actors", ["핵심 채널"])[0], "핵심 채널"),
        "bottom": _short_label(company_name, "운영사"),
    }


def _build_note_texts(source: Dict[str, Any]) -> Dict[str, str]:
    summary = source.get("strategic_summary", {})
    bmc = source.get("business_model_canvas", {})
    return {
        "lt": _keyword_note(summary.get("problem", ""), fallback="보안 위협, 대응 지연"),
        "rt": _keyword_note(_first_item(bmc.get("revenue_streams", [])), fallback="구독료, 라이선스"),
        "lm": _keyword_note(_first_item(bmc.get("value_propositions", [])), fallback="보안 점검, 취약점 탐지"),
        "rm": _keyword_note(_first_item(bmc.get("channels", [])), fallback="직접 영업, 파트너 도입"),
    }


def _build_subtitle(source: Dict[str, Any]) -> str:
    bm_type = clean_korean_label(source.get("bm_type", ""), fallback="플랫폼형")
    if not bm_type.endswith("형"):
        bm_type = f"{bm_type}형"
    return bm_type


def _build_flow_specs(source: Dict[str, Any], node_labels: Dict[str, str]) -> List[Dict[str, Any]]:
    fallback_pairs = [
        ("top", "bottom"),
        ("bottom", "left"),
        ("top", "center"),
        ("left", "center"),
        ("right", "center"),
        ("center", "top"),
        ("center", "right"),
        ("bottom", "center"),
    ]
    flow_groups = [
        (source.get("money_flows", []), "money"),
        (source.get("information_flows", []), "info"),
        (source.get("service_flows", []), "asset"),
    ]

    specs = []
    pair_buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for items, flow_type in flow_groups:
        for idx, item in enumerate(items[: len(fallback_pairs)]):
            start_key = _resolve_node_key(item.get("from", ""), node_labels)
            end_key = _resolve_node_key(item.get("to", ""), node_labels)
            if not start_key or not end_key or start_key == end_key:
                start_key, end_key = fallback_pairs[idx]
            specs.append(
                {
                    "start_key": start_key,
                    "end_key": end_key,
                    "label": _short_flow_label(item.get("label", "")),
                    "type": flow_type,
                }
            )
            pair_buckets[tuple(sorted((start_key, end_key)))].append(len(specs) - 1)

    for pair_indices in pair_buckets.values():
        offsets = _lane_offsets(len(pair_indices))
        for pos, spec_idx in enumerate(pair_indices):
            specs[spec_idx]["lane_offset"] = offsets[pos]
    return specs


def _lane_offsets(count: int) -> List[float]:
    if count == 1:
        return [0.0]
    center = (count - 1) / 2
    return [(idx - center) * 18.0 for idx in range(count)]


def _draw_flow(spec: Dict[str, Any]) -> str:
    start_box = NODE_BOXES[spec["start_key"]]
    end_box = NODE_BOXES[spec["end_key"]]
    sx, sy = _edge_anchor(start_box, end_box)
    ex, ey = _edge_anchor(end_box, start_box)
    lane = spec.get("lane_offset", 0.0)

    if abs(ex - sx) > abs(ey - sy):
        mid_x = (sx + ex) / 2 + lane
        path = f"M {sx} {sy} L {mid_x} {sy} L {mid_x} {ey} L {ex} {ey}"
        mx, my = mid_x, sy
        tx, ty = mid_x, sy - 12
        anchor = "middle"
    else:
        mid_y = (sy + ey) / 2 + lane
        path = f"M {sx} {sy} L {sx} {mid_y} L {ex} {mid_y} L {ex} {ey}"
        mx, my = sx, mid_y
        tx, ty = sx + 18, mid_y - 10
        anchor = "start"

    return (
        f'<path d="{path}" fill="none" stroke="#111827" stroke-width="2.4" marker-end="url(#arrow-black)"/>'
        f'{_flow_marker(mx, my, spec["type"])}'
        f'<text x="{tx}" y="{ty}" text-anchor="{anchor}" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" '
        f'font-size="12.5" font-weight="800" fill="#111827">{html.escape(spec["label"])}</text>'
    )


def _flow_marker(x: float, y: float, flow_type: str) -> str:
    if flow_type == "money":
        return (
            f'<rect x="{x-8}" y="{y-8}" width="16" height="16" rx="4" ry="4" fill="#47b7de" stroke="#111827" stroke-width="1.3"/>'
            f'<text x="{x}" y="{y+4}" text-anchor="middle" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" font-size="10" font-weight="900" fill="#ffffff">W</text>'
        )
    if flow_type == "info":
        return f'<rect x="{x-7}" y="{y-7}" width="14" height="14" fill="#ffffff" stroke="#111827" stroke-width="1.5"/>'
    return f'<circle cx="{x}" cy="{y}" r="7" fill="#d7f0f9" stroke="#111827" stroke-width="1.5"/>'


def _entity_node(node_id: str, x: float, y: float, w: float, h: float, label: str, role: str) -> str:
    kind = _node_kind(label, role)
    cx = x + w / 2
    if kind == "person":
        shape = (
            f'<circle cx="{cx}" cy="{y+24}" r="16" fill="#ffffff" stroke="#111827" stroke-width="2.5"/>'
            f'<path d="M {cx} {y+40} L {cx} {y+82} M {cx-22} {y+56} L {cx+22} {y+56} '
            f'M {cx-24} {y+110} L {cx} {y+82} L {cx+24} {y+110}" fill="none" stroke="#111827" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    elif kind == "platform":
        shape = (
            f'<rect x="{cx-38}" y="{y+4}" width="76" height="104" rx="18" ry="18" fill="#ffffff" stroke="#111827" stroke-width="2.6"/>'
            f'<path d="M {cx-8} {y+24} a 22 22 0 1 1 0.1 0" fill="none" stroke="#47b7de" stroke-width="5"/>'
            f'<path d="M {cx+8} {y+18} l 10 4 l -4 10" fill="none" stroke="#47b7de" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    elif kind == "data":
        shape = (
            f'<rect x="{cx-26}" y="{y+8}" width="52" height="84" fill="#ffffff" stroke="#111827" stroke-width="2.4"/>'
            f'<rect x="{cx-26}" y="{y+8}" width="52" height="44" fill="#a7ddf0" stroke="#111827" stroke-width="2.4"/>'
        )
    elif kind == "asset":
        shape = f'<circle cx="{cx}" cy="{y+50}" r="28" fill="#8fd4ea" stroke="#111827" stroke-width="2.4"/>'
    else:
        shape = (
            f'<rect x="{cx-30}" y="{y+10}" width="60" height="86" fill="#ffffff" stroke="#111827" stroke-width="2.4"/>'
            f'<rect x="{cx-18}" y="{y+22}" width="10" height="10" fill="#ffffff" stroke="#111827" stroke-width="1.1"/>'
            f'<rect x="{cx-2}" y="{y+22}" width="10" height="10" fill="#ffffff" stroke="#111827" stroke-width="1.1"/>'
            f'<rect x="{cx+14}" y="{y+22}" width="10" height="10" fill="#ffffff" stroke="#111827" stroke-width="1.1"/>'
            f'<rect x="{cx-18}" y="{y+40}" width="10" height="10" fill="#ffffff" stroke="#111827" stroke-width="1.1"/>'
            f'<rect x="{cx-2}" y="{y+40}" width="10" height="10" fill="#ffffff" stroke="#111827" stroke-width="1.1"/>'
            f'<rect x="{cx+14}" y="{y+40}" width="10" height="10" fill="#ffffff" stroke="#111827" stroke-width="1.1"/>'
        )
    return f'<g id="{node_id}">{shape}{_node_label(cx, y+h-8, label)}</g>'


def _company_node(node_id: str, x: float, y: float, w: float, h: float, label: str) -> str:
    cx = x + w / 2
    bx = cx - 30
    by = y
    return (
        f'<g id="{node_id}">'
        f'<rect x="{bx}" y="{by}" width="60" height="88" fill="#ffffff" stroke="#111827" stroke-width="2.8"/>'
        f'<rect x="{bx+10}" y="{by+10}" width="10" height="10" fill="#ffffff" stroke="#111827" stroke-width="1.3"/>'
        f'<rect x="{bx+24}" y="{by+10}" width="10" height="10" fill="#ffffff" stroke="#111827" stroke-width="1.3"/>'
        f'<rect x="{bx+38}" y="{by+10}" width="10" height="10" fill="#ffffff" stroke="#111827" stroke-width="1.3"/>'
        f'<rect x="{bx+10}" y="{by+28}" width="10" height="10" fill="#ffffff" stroke="#111827" stroke-width="1.3"/>'
        f'<rect x="{bx+24}" y="{by+28}" width="10" height="10" fill="#ffffff" stroke="#111827" stroke-width="1.3"/>'
        f'<rect x="{bx+38}" y="{by+28}" width="10" height="10" fill="#ffffff" stroke="#111827" stroke-width="1.3"/>'
        f'<rect x="{bx+24}" y="{by+50}" width="12" height="24" fill="#ffffff" stroke="#111827" stroke-width="1.3"/>'
        f'{_node_label(cx, y+h-6, label)}'
        '</g>'
    )


def _node_label(cx: float, y: float, label: str) -> str:
    lines = _wrap_text(label, max_chars=10, max_lines=2)
    return _multiline_text(cx, y, lines, 17, "#3aa7ca")


def _note_box(x: float, y: float, w: float, h: float, body: str) -> str:
    lines = _wrap_text(body, max_chars=12, max_lines=3)
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#edf8fc" stroke="none"/>'
        f'{_multiline_text(x+8, y+20, lines, 11.5, "#60707c", weight="700", anchor="start")}'
    )


def _multiline_text(x: float, y: float, lines: List[str], font_size: float, fill: str, weight: str = "800", anchor: str = "middle") -> str:
    line_h = font_size + 5
    tspans = []
    for idx, line in enumerate(lines):
        tspans.append(f'<tspan x="{x}" y="{y + idx * line_h}">{html.escape(line)}</tspan>')
    return (
        f'<text text-anchor="{anchor}" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" '
        f'font-size="{font_size}" font-weight="{weight}" fill="{fill}">' + "".join(tspans) + "</text>"
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


def _resolve_node_key(name: str, node_labels: Dict[str, str]) -> str:
    text = clean_korean_label(name)
    if not text:
        return ""
    for key, label in node_labels.items():
        if text == label:
            return key
    for key, label in node_labels.items():
        if text in label or label in text:
            return key
    return ""


def _short_label(text: Any, fallback: str) -> str:
    value = clean_korean_label(text, fallback=fallback)
    replacements = {
        "주요 금융기관": "금융기관",
        "LLM 기반 서비스 기업": "AI 서비스사",
        "클라우드 서비스 제공사": "기술 공급사",
        "AI 기술 제공 업체": "기술 공급사",
        "보안 솔루션 유통 파트너": "유통 파트너",
        "직접 판매": "직접 고객",
        "세이프엑스(SAIFEX)": "세이프엑스",
        "세이프엑스(SAIFE X)": "세이프엑스",
    }
    value = replacements.get(value, value)
    value = value.replace("서비스", "").replace("플랫폼", "").replace("솔루션", "")
    value = value.strip()
    if not value:
        value = fallback
    return value


def _short_flow_label(text: Any) -> str:
    value = clean_korean_label(text, fallback="흐름")
    replacements = {
        "플랫폼 운영": "운영 지원",
        "운영 데이터": "운영 데이터",
        "요청 정보": "보안 요청",
        "사용 데이터": "사용 데이터",
        "도입 정보": "도입 정보",
        "채널 정보": "채널 정보",
        "기술 연동": "기술 연동",
        "솔루션 제공": "서비스 제공",
        "채널 지원": "채널 지원",
        "보안 서비스": "분석 결과",
        "모델 비용": "모델 비용",
        "인프라 비용": "인프라 비용",
        "구독료": "구독료",
    }
    return replacements.get(value, value[:10])


def _keyword_note(text: Any, fallback: str) -> str:
    value = clean_korean_label(text, fallback=fallback)
    replacements = [
        (" 및 ", ", "),
        (" 와 ", ", "),
        (" 그리고 ", ", "),
        (" 생성형 AI ", "생성형AI "),
        (" 취약점 ", "취약점 "),
        (" 대응 ", "대응 "),
    ]
    for old, new in replacements:
        value = value.replace(old, new)
    value = value.replace(" 해결", "").replace(" 제공", "").replace(" 구조", "")
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        parts = [fallback]
    short_parts = []
    for part in parts[:3]:
        cleaned = part[:12].strip()
        if cleaned:
            short_parts.append(cleaned)
    return ", ".join(short_parts[:3])


def _first_item(values: List[str]) -> str:
    for value in values or []:
        text = clean_korean_label(value)
        if text:
            return text
    return ""


def _wrap_text(text: str, max_chars: int = 10, max_lines: int = 2) -> List[str]:
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


def _node_kind(label: str, role: str) -> str:
    text = clean_korean_label(label)
    if role == "center":
        return "platform"
    if role == "top":
        if any(token in text for token in ["기관", "기업", "회사", "고객사", "은행"]):
            return "company"
        return "person"
    if any(token in text for token in ["데이터", "정보", "점수", "토큰"]):
        return "data"
    if any(token in text for token in ["상품", "물건", "자산", "펀드", "중고품"]):
        return "asset"
    return "company"
