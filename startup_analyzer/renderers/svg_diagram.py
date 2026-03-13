import html
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import streamlit.components.v1 as components

from startup_analyzer.services.bmc import ensure_bmc_shape
from startup_analyzer.utils.text import clean_korean_label


CANVAS_WIDTH = 1120
CANVAS_HEIGHT = 760
FRAME_X = 40
FRAME_Y = 24
FRAME_W = 1040
FRAME_H = 680

NODE_BOXES = {
    "top": (460, 118, 200, 84),
    "left": (108, 286, 190, 92),
    "center": (438, 252, 244, 164),
    "right": (822, 286, 190, 92),
    "bottom": (454, 520, 212, 92),
}

NOTE_BOXES = {
    "left_top": (70, 116, 150, 78),
    "right_top": (900, 116, 150, 78),
    "left_bottom": (76, 514, 180, 86),
    "right_bottom": (860, 514, 190, 86),
}


def render_svg_preview(svg_bytes: bytes, height: int = 780):
    components.html(svg_bytes.decode("utf-8"), height=height, scrolling=True)


def build_editable_svg(data: Dict[str, Any], company_name: str) -> bytes:
    source = ensure_bmc_shape(data, company_name=company_name)
    top_label = _first_non_empty(source.get("top_layer", []), "핵심 고객")
    left_label = _first_non_empty(source.get("left_actors", []), "핵심 파트너")
    center_label = clean_korean_label(source.get("middle_layer", ""), fallback="핵심 플랫폼")
    right_label = _first_non_empty(source.get("right_actors", []), "핵심 채널")
    bottom_label = clean_korean_label(company_name, fallback=company_name)

    node_labels = {
        "top": top_label,
        "left": left_label,
        "center": center_label,
        "right": right_label,
        "bottom": bottom_label,
    }

    flow_specs = _build_flow_specs(source, node_labels)
    flow_fragments = [_orthogonal_flow(item) for item in flow_specs]

    notes = _build_notes(source)
    note_fragments = [
        _note_box(*NOTE_BOXES["left_top"], "핵심 가치", notes["value"]),
        _note_box(*NOTE_BOXES["right_top"], "수익 구조", notes["revenue"]),
        _note_box(*NOTE_BOXES["left_bottom"], "운영 포인트", notes["operation"]),
        _note_box(*NOTE_BOXES["right_bottom"], "핵심 자원", notes["resource"]),
    ]

    section_lines = [
        _section_line(228),
        _section_line(448),
    ]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">
<defs>
  <marker id="arrow-green" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#4b9e58"/>
  </marker>
  <marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#3b82db"/>
  </marker>
  <marker id="arrow-orange" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#e58b2c"/>
  </marker>
</defs>
<rect x="0" y="0" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" fill="#ffffff"/>
<path d="M {FRAME_X+18} {FRAME_Y} H {FRAME_X+FRAME_W-18} Q {FRAME_X+FRAME_W} {FRAME_Y} {FRAME_X+FRAME_W} {FRAME_Y+18} V {FRAME_Y+54} H {FRAME_X} V {FRAME_Y+18} Q {FRAME_X} {FRAME_Y} {FRAME_X+18} {FRAME_Y} Z" fill="#4bb6df"/>
<rect x="{FRAME_X}" y="{FRAME_Y}" width="{FRAME_W}" height="{FRAME_H}" rx="26" ry="26" fill="none" stroke="#4bb6df" stroke-width="4"/>
<text x="{FRAME_X+24}" y="{FRAME_Y+40}" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" font-size="32" font-weight="800" fill="#1d2630">{html.escape(bottom_label)}</text>
<text x="{FRAME_X+24}" y="{FRAME_Y+66}" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" font-size="15" font-weight="600" fill="#1d2630">비즈니스모델 다이어그램</text>
{''.join(section_lines)}
{''.join(note_fragments)}
{''.join(flow_fragments)}
{_node_actor("node-top", *NODE_BOXES["top"], top_label, "사용자")}
{_node_block("node-left", *NODE_BOXES["left"], left_label, "연동")}
{_node_platform("node-center", *NODE_BOXES["center"], center_label)}
{_node_block("node-right", *NODE_BOXES["right"], right_label, "채널")}
{_node_company("node-bottom", *NODE_BOXES["bottom"], bottom_label)}
{_legend()}
</svg>"""
    return svg.encode("utf-8")


def _build_notes(source: Dict[str, Any]) -> Dict[str, str]:
    bmc = source.get("business_model_canvas", {})
    return {
        "value": _note_text(bmc.get("value_propositions", []), "AI 보안 분석 자동화"),
        "revenue": _note_text(bmc.get("revenue_streams", []), "구독형 과금"),
        "operation": _note_text(bmc.get("key_activities", []), "플랫폼 운영"),
        "resource": _note_text(bmc.get("key_resources", []), "AI 분석 엔진"),
    }


def _note_text(values: List[str], fallback: str) -> str:
    if values:
        return clean_korean_label(values[0], fallback=fallback)
    return fallback


def _build_flow_specs(source: Dict[str, Any], node_labels: Dict[str, str]) -> List[Dict[str, Any]]:
    fallback_pairs = [
        ("top", "bottom"),
        ("bottom", "left"),
        ("top", "center"),
        ("left", "center"),
        ("center", "bottom"),
        ("right", "center"),
        ("center", "top"),
        ("bottom", "center"),
        ("center", "right"),
    ]

    specs = []
    pair_buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    flow_groups = [
        (source.get("money_flows", []), "#4b9e58", "arrow-green", "W"),
        (source.get("information_flows", []), "#3b82db", "arrow-blue", "i"),
        (source.get("service_flows", []), "#e58b2c", "arrow-orange", "S"),
    ]

    for items, color, marker_id, badge in flow_groups:
        for idx, item in enumerate(items[: len(fallback_pairs)]):
            start_key = _resolve_node_key(item.get("from", ""), node_labels)
            end_key = _resolve_node_key(item.get("to", ""), node_labels)
            if not start_key or not end_key or start_key == end_key:
                start_key, end_key = fallback_pairs[idx]
            spec = {
                "start_key": start_key,
                "end_key": end_key,
                "label": clean_korean_label(item.get("label", ""), fallback="흐름"),
                "color": color,
                "marker_id": marker_id,
                "badge": badge,
            }
            specs.append(spec)
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


def _orthogonal_flow(item: Dict[str, Any]) -> str:
    start_box = NODE_BOXES[item["start_key"]]
    end_box = NODE_BOXES[item["end_key"]]
    sx, sy = _edge_anchor(start_box, end_box)
    ex, ey = _edge_anchor(end_box, start_box)
    lane = item.get("lane_offset", 0.0)

    if abs(ex - sx) > abs(ey - sy):
        mid_x = (sx + ex) / 2 + lane
        path = f"M {sx} {sy} L {mid_x} {sy} L {mid_x} {ey} L {ex} {ey}"
        lx, ly = mid_x, sy - 14
    else:
        mid_y = (sy + ey) / 2 + lane
        path = f"M {sx} {sy} L {sx} {mid_y} L {ex} {mid_y} L {ex} {ey}"
        lx, ly = sx + 14, mid_y - 10

    return (
        f'<path d="{path}" fill="none" stroke="{item["color"]}" stroke-width="3.5" marker-end="url(#{item["marker_id"]})"/>'
        f'{_flow_badge(lx, ly, item["badge"], item["label"], item["color"])}'
    )


def _flow_badge(x: float, y: float, badge: str, label: str, color: str) -> str:
    text = clean_korean_label(label, fallback="흐름")
    width = max(66, 18 + len(text) * 12)
    return (
        f'<g transform="translate({x},{y})">'
        f'<rect x="-12" y="-12" width="22" height="22" rx="6" ry="6" fill="{color}"/>'
        f'<text x="-1" y="4" text-anchor="middle" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" font-size="12" font-weight="800" fill="#ffffff">{html.escape(badge)}</text>'
        f'<rect x="16" y="-14" width="{width}" height="26" rx="8" ry="8" fill="#ffffff" stroke="{color}" stroke-width="2"/>'
        f'<text x="{16 + width/2}" y="4" text-anchor="middle" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" font-size="14" font-weight="700" fill="#1f2937">{html.escape(text)}</text>'
        '</g>'
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


def _section_line(y: float) -> str:
    return f'<line x1="{FRAME_X+26}" y1="{y}" x2="{FRAME_X+FRAME_W-26}" y2="{y}" stroke="#d6e7ef" stroke-width="2"/>'


def _wrap_svg_text(text: str, max_chars: int = 12, max_lines: int = 2) -> List[str]:
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


def _multiline_text(x: float, y: float, lines: List[str], font_size: int, fill: str, weight: str = "700", anchor: str = "middle") -> str:
    line_h = font_size + 6
    tspans = []
    for idx, line in enumerate(lines):
        tspans.append(f'<tspan x="{x}" y="{y + idx * line_h}">{html.escape(line)}</tspan>')
    return (
        f'<text text-anchor="{anchor}" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" '
        f'font-size="{font_size}" font-weight="{weight}" fill="{fill}">' + "".join(tspans) + "</text>"
    )


def _node_actor(node_id: str, x: float, y: float, w: float, h: float, label: str, caption: str) -> str:
    return (
        f'<g id="{node_id}">'
        f'<circle cx="{x + w/2}" cy="{y + 16}" r="12" fill="none" stroke="#1f2937" stroke-width="2.5"/>'
        f'<path d="M {x + w/2} {y + 28} L {x + w/2} {y + 58} M {x + w/2 - 18} {y + 42} L {x + w/2 + 18} {y + 42} M {x + w/2 - 18} {y + 82} L {x + w/2} {y + 58} L {x + w/2 + 18} {y + 82}" fill="none" stroke="#1f2937" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
        f'{_multiline_text(x + w/2, y + 108, _wrap_svg_text(label, max_chars=10), 22, "#1395c3")}'
        f'{_multiline_text(x + w/2, y + 132, [caption], 13, "#64748b", weight="600")}'
        '</g>'
    )


def _node_block(node_id: str, x: float, y: float, w: float, h: float, label: str, caption: str) -> str:
    return (
        f'<g id="{node_id}">'
        f'<rect x="{x+28}" y="{y}" width="{w-56}" height="{h}" rx="10" ry="10" fill="#e8f7fd" stroke="#7fc9e4" stroke-width="2"/>'
        f'<rect x="{x+52}" y="{y+18}" width="{w-104}" height="22" fill="#7fc9e4" opacity="0.65"/>'
        f'<rect x="{x+52}" y="{y+44}" width="{w-104}" height="22" fill="#ffffff" stroke="#7fc9e4" stroke-width="1"/>'
        f'{_multiline_text(x + w/2, y + h + 28, _wrap_svg_text(label, max_chars=10), 20, "#1395c3")}'
        f'{_multiline_text(x + w/2, y + h + 50, [caption], 13, "#64748b", weight="600")}'
        '</g>'
    )


def _node_platform(node_id: str, x: float, y: float, w: float, h: float, label: str) -> str:
    return (
        f'<g id="{node_id}">'
        f'<rect x="{x+74}" y="{y}" width="{w-148}" height="{h}" rx="18" ry="18" fill="#ffffff" stroke="#1f2937" stroke-width="3"/>'
        f'<rect x="{x+98}" y="{y+24}" width="{w-196}" height="{h-62}" rx="12" ry="12" fill="#e8f7fd" stroke="#7fc9e4" stroke-width="2"/>'
        f'<circle cx="{x+w/2}" cy="{y+h-18}" r="6" fill="#1f2937"/>'
        f'{_multiline_text(x + w/2, y + h + 28, _wrap_svg_text(label, max_chars=12), 24, "#1395c3")}'
        f'{_multiline_text(x + w/2, y + h + 54, ["플랫폼"], 13, "#64748b", weight="600")}'
        '</g>'
    )


def _node_company(node_id: str, x: float, y: float, w: float, h: float, label: str) -> str:
    return (
        f'<g id="{node_id}">'
        f'<rect x="{x+68}" y="{y}" width="{w-136}" height="{h}" rx="8" ry="8" fill="#ffffff" stroke="#1f2937" stroke-width="3"/>'
        f'<path d="M {x+88} {y+18} h {w-176} M {x+88} {y+40} h {w-176} M {x+88} {y+62} h {w-176}" stroke="#7fc9e4" stroke-width="8"/>'
        f'{_multiline_text(x + w/2, y + h + 28, _wrap_svg_text(label, max_chars=12), 22, "#1395c3")}'
        f'{_multiline_text(x + w/2, y + h + 52, ["운영사"], 13, "#64748b", weight="600")}'
        '</g>'
    )


def _note_box(x: float, y: float, w: float, h: float, title: str, body: str) -> str:
    lines = _wrap_svg_text(body, max_chars=13, max_lines=3)
    return (
        f'<g>'
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" ry="12" fill="#f8fbfd" stroke="#cbe6f1" stroke-width="2"/>'
        f'{_multiline_text(x + 14, y + 24, [title], 13, "#4b5c74", weight="800", anchor="start")}'
        f'{_multiline_text(x + 14, y + 48, lines, 14, "#334155", weight="600", anchor="start")}'
        '</g>'
    )


def _legend() -> str:
    return (
        '<g transform="translate(350,664)">'
        '<rect x="-12" y="-4" width="420" height="34" rx="14" ry="14" fill="#f8fbfd" stroke="#cbe6f1" stroke-width="2"/>'
        '<text x="10" y="18" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" font-size="13" font-weight="800" fill="#4b5c74">흐름:</text>'
        '<text x="62" y="18" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" font-size="13" font-weight="700" fill="#4b9e58">W 돈</text>'
        '<text x="136" y="18" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" font-size="13" font-weight="700" fill="#3b82db">i 정보</text>'
        '<text x="214" y="18" font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" font-size="13" font-weight="700" fill="#e58b2c">S 서비스/운영</text>'
        '</g>'
    )


def _first_non_empty(values: Any, fallback: str = "") -> str:
    if isinstance(values, list):
        for value in values:
            text = clean_korean_label(value)
            if text:
                return text
        return fallback
    text = clean_korean_label(values)
    return text or fallback


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
