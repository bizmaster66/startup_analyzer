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

SUMMARY_BAR = (170, 656, 640, 46)


def render_svg_preview(svg_bytes: bytes, height: int = 780):
    components.html(svg_bytes.decode("utf-8"), height=height, scrolling=True)


def build_editable_svg(data: Dict[str, Any], company_name: str) -> bytes:
    source = ensure_bmc_shape(data, company_name=company_name)
    node_labels = _build_node_labels(source, company_name)
    note_texts = _build_note_texts(source)
    summary = _build_summary_texts(source)
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
{_header(company_name, summary["subtitle"])}
{_section_lines()}
{''.join(note_fragments)}
{''.join(flow_fragments)}
{_person_node("top", *NODE_BOXES["top"], node_labels["top"])}
{_block_node("left", *NODE_BOXES["left"], node_labels["left"], _node_kind(node_labels["left"]))}
{_phone_node("center", *NODE_BOXES["center"], node_labels["center"])}
{_block_node("right", *NODE_BOXES["right"], node_labels["right"], _node_kind(node_labels["right"]))}
{_company_node("bottom", *NODE_BOXES["bottom"], node_labels["bottom"])}
{_summary_bar(summary)}
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
        "lt": _note_snippet(summary.get("problem", ""), fallback="고객 문제를 간결하게 해결"),
        "rt": _note_snippet(_first_item(bmc.get("revenue_streams", [])), fallback="반복 매출 구조 확보"),
        "lm": _note_snippet(_first_item(bmc.get("value_propositions", [])), fallback="핵심 가치 제공"),
        "rm": _note_snippet(_first_item(bmc.get("channels", [])), fallback="도입과 유통을 지원"),
    }


def _build_summary_texts(source: Dict[str, Any]) -> Dict[str, str]:
    summary = source.get("strategic_summary", {})
    bmc = source.get("business_model_canvas", {})
    topic = _short_label(_first_item(source.get("business_model_canvas", {}).get("customer_segments", [])), "핵심 문제")
    subtitle = _note_snippet(source.get("bmc_summary", ""), fallback=_first_item(bmc.get("value_propositions", [])) or "핵심 가치로 시장 문제를 해결")
    return {
        "topic": topic,
        "정설": _note_snippet(summary.get("status_quo", ""), fallback="기존 방식은 느리고 비효율적이다"),
        "역설": _note_snippet(summary.get("our_solution", ""), fallback="플랫폼으로 빠르게 해결한다"),
        "subtitle": subtitle,
    }


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
        f'font-size="13" font-weight="800" fill="#111827">{html.escape(spec["label"])}</text>'
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


def _person_node(node_id: str, x: float, y: float, w: float, h: float, label: str) -> str:
    cx = x + w / 2
    return (
        f'<g id="{node_id}">'
        f'<circle cx="{cx}" cy="{y+22}" r="16" fill="#ffffff" stroke="#111827" stroke-width="2.5"/>'
        f'<path d="M {cx} {y+38} L {cx} {y+82} M {cx-22} {y+54} L {cx+22} {y+54} M {cx-24} {y+110} L {cx} {y+82} L {cx+24} {y+110}" fill="none" stroke="#111827" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
        f'{_node_label(cx, y+h-8, label)}'
        '</g>'
    )


def _phone_node(node_id: str, x: float, y: float, w: float, h: float, label: str) -> str:
    cx = x + w / 2
    return (
        f'<g id="{node_id}">'
        f'<rect x="{cx-30}" y="{y}" width="60" height="108" rx="12" ry="12" fill="#ffffff" stroke="#111827" stroke-width="2.8"/>'
        f'<rect x="{cx-20}" y="{y+18}" width="40" height="64" fill="#f3f8fb" stroke="#111827" stroke-width="1.4"/>'
        f'<circle cx="{cx}" cy="{y+92}" r="4" fill="#111827"/>'
        f'{_node_label(cx, y+h-6, label)}'
        '</g>'
    )


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


def _block_node(node_id: str, x: float, y: float, w: float, h: float, label: str, kind: str) -> str:
    cx = x + w / 2
    if kind == "정보":
        shape = (
            f'<rect x="{cx-24}" y="{y+6}" width="48" height="86" fill="#ffffff" stroke="#111827" stroke-width="2.6"/>'
            f'<rect x="{cx-24}" y="{y+6}" width="48" height="42" fill="#bfe9f7" stroke="#111827" stroke-width="2.6"/>'
        )
    elif kind == "물건":
        shape = f'<circle cx="{cx}" cy="{y+48}" r="26" fill="#8fd4ea" stroke="#111827" stroke-width="2.6"/>'
    else:
        shape = (
            f'<rect x="{cx-26}" y="{y+6}" width="52" height="84" fill="#ffffff" stroke="#111827" stroke-width="2.6"/>'
            f'<rect x="{cx-16}" y="{y+16}" width="10" height="10" fill="#ffffff" stroke="#111827" stroke-width="1.2"/>'
            f'<rect x="{cx}" y="{y+16}" width="10" height="10" fill="#ffffff" stroke="#111827" stroke-width="1.2"/>'
            f'<rect x="{cx-16}" y="{y+32}" width="10" height="10" fill="#ffffff" stroke="#111827" stroke-width="1.2"/>'
            f'<rect x="{cx}" y="{y+32}" width="10" height="10" fill="#ffffff" stroke="#111827" stroke-width="1.2"/>'
        )
    return f'<g id="{node_id}">{shape}{_node_label(cx, y+h-6, label)}</g>'


def _node_label(cx: float, y: float, label: str) -> str:
    lines = _wrap_text(label, max_chars=10, max_lines=2)
    return _multiline_text(cx, y, lines, 17, "#3aa7ca")


def _note_box(x: float, y: float, w: float, h: float, body: str) -> str:
    lines = _wrap_text(body, max_chars=12, max_lines=4)
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#edf8fc" stroke="none"/>'
        f'{_multiline_text(x+8, y+18, lines, 11.5, "#60707c", weight="700", anchor="start")}'
    )


def _summary_bar(summary: Dict[str, str]) -> str:
    x, y, w, h = SUMMARY_BAR
    return (
        f'<text x="{x-88}" y="{y+30}" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" font-size="16" font-weight="800" fill="#111827">{html.escape(summary["topic"])}</text>'
        f'<rect x="{x}" y="{y}" width="62" height="{h}" fill="#1f8fb6"/>'
        f'<text x="{x+31}" y="{y+29}" text-anchor="middle" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" font-size="14" font-weight="800" fill="#ffffff">기점</text>'
        f'<rect x="{x+72}" y="{y}" width="62" height="{h}" fill="#2a9cc2"/>'
        f'<text x="{x+103}" y="{y+19}" text-anchor="middle" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" font-size="13" font-weight="800" fill="#ffffff">정설</text>'
        f'<text x="{x+150}" y="{y+18}" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" font-size="13" font-weight="800" fill="#111827">{html.escape(summary["정설"])}</text>'
        f'<rect x="{x+72}" y="{y+24}" width="62" height="22" fill="#2a9cc2"/>'
        f'<text x="{x+103}" y="{y+40}" text-anchor="middle" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" font-size="13" font-weight="800" fill="#ffffff">역설</text>'
        f'<text x="{x+150}" y="{y+40}" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" font-size="13" font-weight="800" fill="#111827">{html.escape(summary["역설"])}</text>'
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
    value = value.replace("서비스", "").replace("플랫폼", "").replace("솔루션", "")
    value = value.strip()
    if not value:
        value = fallback
    return value


def _short_flow_label(text: Any) -> str:
    value = clean_korean_label(text, fallback="흐름")
    replacements = {
        "플랫폼 운영": "운영",
        "운영 데이터": "운영",
        "요청 정보": "정보",
        "사용 데이터": "정보",
        "도입 정보": "도입",
        "기술 연동": "연동",
        "솔루션 제공": "제공",
        "보안 서비스": "사용",
        "모델 비용": "비용",
        "인프라 비용": "비용",
    }
    return replacements.get(value, value[:8])


def _note_snippet(text: Any, fallback: str) -> str:
    value = clean_korean_label(text, fallback=fallback)
    value = value.replace("  ", " ").strip()
    if len(value) > 32:
        value = value[:32].rstrip() + "..."
    return value


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


def _node_kind(label: str) -> str:
    text = clean_korean_label(label)
    if any(token in text for token in ["데이터", "정보", "점수", "토큰"]):
        return "정보"
    if any(token in text for token in ["상품", "물건", "펀드", "자산", "토큰", "중고품"]):
        return "물건"
    return "회사"
