import html
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import streamlit.components.v1 as components

from startup_analyzer.services.bmc import ensure_bmc_shape
from startup_analyzer.utils.text import clean_korean_label


NODE_BOXES = {
    "top": (620, 90, 260, 100),
    "left": (70, 335, 280, 112),
    "center": (635, 325, 270, 124),
    "right": (1250, 335, 280, 112),
    "bottom": (620, 610, 280, 100),
}


def render_svg_preview(svg_bytes: bytes, height: int = 920):
    components.html(svg_bytes.decode("utf-8"), height=height, scrolling=True)


def build_editable_svg(data: Dict[str, Any], company_name: str) -> bytes:
    source = ensure_bmc_shape(data, company_name=company_name)
    width, height = 1600, 900

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
    flow_fragments = [
        _svg_flow_path(
            NODE_BOXES[item["start_key"]],
            NODE_BOXES[item["end_key"]],
            item["color"],
            item["marker_id"],
            item["label"],
            item["symbol"],
            item["lane_offset"],
            item["direction_bias"],
        )
        for item in flow_specs
    ]

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

    title = f"{html.escape(bottom_label)} BM 다이어그램"
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
{_svg_rect_node("node-top", *NODE_BOXES["top"], top_label, "◌")}
{_svg_rect_node("node-left", *NODE_BOXES["left"], left_label, "◌")}
{_svg_rect_node("node-center", *NODE_BOXES["center"], center_label, "◎")}
{_svg_rect_node("node-right", *NODE_BOXES["right"], right_label, "◌")}
{_svg_rect_node("node-bottom", *NODE_BOXES["bottom"], bottom_label, "◌")}
{legend_frag}
</svg>"""
    return svg.encode("utf-8")


def _build_flow_specs(source: Dict[str, Any], node_labels: Dict[str, str]) -> List[Dict[str, Any]]:
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

    specs = []
    pair_buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    for items, color, marker_id, symbol in [
        (source.get("money_flows", []), "#4b9e58", "arrow-green", "$"),
        (source.get("information_flows", []), "#3b82db", "arrow-blue", "□"),
        (source.get("service_flows", []), "#e58b2c", "arrow-orange", "○"),
    ]:
        for idx, item in enumerate(items[: len(fallback_pairs)]):
            start_key = _resolve_node_key(item.get("from", ""), node_labels)
            end_key = _resolve_node_key(item.get("to", ""), node_labels)
            if not start_key or not end_key or start_key == end_key:
                start_key, end_key = fallback_pairs[idx]

            label = clean_korean_label(item.get("label", ""), fallback="흐름")
            spec = {
                "start_key": start_key,
                "end_key": end_key,
                "label": label,
                "color": color,
                "marker_id": marker_id,
                "symbol": symbol,
            }
            specs.append(spec)
            pair_buckets[tuple(sorted((start_key, end_key)))].append(len(specs) - 1)

    for pair_indices in pair_buckets.values():
        offsets = _lane_offsets(len(pair_indices))
        for pos, spec_idx in enumerate(pair_indices):
            spec = specs[spec_idx]
            spec["lane_offset"] = offsets[pos]
            spec["direction_bias"] = 1 if spec["start_key"] < spec["end_key"] else -1

    return specs


def _lane_offsets(count: int) -> List[float]:
    if count == 1:
        return [0.0]

    offsets = []
    center = (count - 1) / 2
    for idx in range(count):
        offsets.append((idx - center) * 34.0)
    return offsets


def _wrap_svg_text(text: str, max_chars: int = 14, max_lines: int = 2) -> List[str]:
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


def _svg_multiline_text(x: float, y: float, w: float, h: float, text: str, font_size: int = 24) -> str:
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
        f'font-size="{font_size}" font-weight="700" fill="#253240">'
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


def _svg_flow_path(
    start_box: tuple,
    end_box: tuple,
    color: str,
    marker_id: str,
    label: str,
    symbol: str,
    lane_offset: float,
    direction_bias: int,
) -> str:
    sx, sy = _edge_anchor(start_box, end_box)
    ex, ey = _edge_anchor(end_box, start_box)
    mx = (sx + ex) / 2
    my = (sy + ey) / 2
    dx, dy = ex - sx, ey - sy

    if abs(dx) > abs(dy):
        cy = my - 56 + lane_offset if dy >= 0 else my + 56 - lane_offset
        cy += direction_bias * 10
        path = f"M {sx} {sy} C {mx} {sy + lane_offset * 0.2}, {mx} {cy}, {ex} {ey}"
        lx, ly = mx, cy - 12
    else:
        cx = mx - 70 + lane_offset if dx >= 0 else mx + 70 - lane_offset
        cx += direction_bias * 12
        path = f"M {sx} {sy} C {sx + lane_offset * 0.2} {my}, {cx} {my}, {ex} {ey}"
        lx, ly = cx, my - 12 + lane_offset * 0.15

    text_label = f"{symbol} {label}".strip()
    return (
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="4" marker-end="url(#{marker_id})"/>'
        f'<text x="{lx}" y="{ly}" text-anchor="middle" '
        'font-family="Apple SD Gothic Neo, NanumGothic, Noto Sans CJK KR, sans-serif" '
        f'font-size="20" font-weight="600" fill="{color}">{html.escape(text_label)}</text>'
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
