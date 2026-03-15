import json
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from startup_analyzer.services.analysis import TEXT_MODEL
from startup_analyzer.utils.json_utils import extract_json, repair_json_with_model
from startup_analyzer.utils.text import clean_korean_label


IMAGE_MODEL = "gemini-3.1-flash-image-preview"


def generate_bm_diagram_png(
    client: genai.Client,
    company_name: str,
    bmc_data: Dict[str, Any],
) -> bytes:
    validated_flows = _validate_role_flows(client, company_name, bmc_data)
    prompt = _build_diagram_prompt(company_name, bmc_data, validated_flows)
    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=[prompt],
    )

    for part in getattr(response, "parts", []) or []:
        inline_data = getattr(part, "inline_data", None)
        if inline_data and getattr(inline_data, "data", None):
            return inline_data.data

    raise ValueError("Gemini 이미지 생성 응답에서 PNG 데이터를 찾지 못했습니다.")


def _build_diagram_prompt(company_name: str, bmc_data: Dict[str, Any], validated_flows: List[Dict[str, str]]) -> str:
    bmc = bmc_data.get("business_model_canvas", {}) or {}

    market_needs = _join_items(bmc.get("customer_relationships", []), fallback="시장 니즈")
    target_users = _join_items(bmc.get("customer_segments", []), fallback="핵심 고객")
    community_channels = _join_items(bmc.get("channels", []), fallback="고객 채널")
    value_prop = _join_items(bmc.get("value_propositions", []), fallback="제공 가치")
    core_platform = clean_korean_label(bmc_data.get("middle_layer", ""), fallback=f"{company_name} 플랫폼")
    activities = _join_items(bmc.get("key_activities", []), fallback="핵심 활동")
    partners = _join_items(bmc.get("key_partnerships", []), fallback="핵심 파트너")
    company = company_name
    resources = _join_items(bmc.get("key_resources", []), fallback="핵심 자원/경쟁력")

    info_flows = _join_flow_labels(bmc_data.get("information_flows", []), fallback="사용 데이터, 도입 정보")
    money_flows = _join_flow_labels(bmc_data.get("money_flows", []), fallback="구독 매출, 제휴 수수료")
    service_flows = _join_flow_labels(bmc_data.get("service_flows", []), fallback="분석 결과, 인프라 제공")
    validated_flow_lines = _format_validated_flows(validated_flows)

    return f"""
한국어 비즈니스 생태계 다이어그램 PNG를 생성하라.

[목표]
- 투자자 및 컨설팅 슬라이드에 즉시 넣을 수 있는 전문가 수준의 Business Model Canvas 3x3 다이어그램
- 복잡한 장식이나 실험적인 플로우차트를 절대 금지하며, 깔끔하고 정돈된 BI 대시보드 형태를 유지할 것

[시각 스타일]
- 배경: 흰색 또는 아주 연한 회색
- 카드: 부드러운 그림자가 있는 rounded rectangle
- 테마: 옅은 파스텔 블루/그레이 카드 배경, 진한 네이비 텍스트
- 아이콘: 각 카드 내용을 직관적으로 나타내는 세련된 2D flat vector 아이콘 또는 이모지 1개를 작게 배치
- 폰트/텍스트: 기존보다 더 작고 컴팩트하게, 현재 수준의 약 50% 체감 크기까지 줄여 여백을 충분히 확보할 것
- 각 노드는 title 1줄, subtitle 1줄, bullet 2~3개만 포함
- 정보 흐름은 파란색 + 사각형 마커
- 돈 흐름은 초록색 + 달러 마커
- 서비스 흐름은 주황색 + 원형 마커
- 글자는 모두 한국어
- 전체 화면 비율은 가로형 16:10에 가깝게
- 하단 중앙에 범례를 반드시 포함
- 범례 텍스트는 정확히 다음 구조를 따를 것:
  범례: $ = 돈 흐름   □ = 정보 흐름   ○ = 서비스/자산 흐름
- 범례에서도 초록색 달러, 파란색 사각형, 주황색 원형 마커를 실제로 그릴 것
- 범례는 다이어그램 본체와 겹치지 않게 하단 여백에 배치할 것
- 별도 메인 타이틀이나 상단 제목 텍스트를 넣지 말 것
- 회사명으로 만든 장식성 제목, 따옴표 제목, 부정확한 헤더 문구를 넣지 말 것
- 출력 해상도는 1k급으로 생성할 것
- 가로형 기준 약 1024px 너비의 선명한 PNG로 생성할 것
- 저해상도, 흐릿한 텍스트, 압축 artifacts 금지

[레이아웃 및 배치 (엄격한 3x3 Grid)]
- 반드시 가로 3칸, 세로 3칸의 균형 잡힌 그리드 구조
- 상단(이용자 레이어): 좌측[시장 상황 및 니즈], 중앙[타겟 고객], 우측[커뮤니티 및 채널]
- 중단(사업 레이어): 좌측[제공 가치], 중앙[코어 플랫폼], 우측[핵심 활동]
- 하단(사업자 레이어): 좌측[핵심 파트너], 중앙[기업 본체], 우측[핵심 자원]
- 중앙의 코어 플랫폼 카드는 다른 카드보다 약간 크거나 시각적으로 더 돋보이게 처리
- 별도 메인 타이틀이나 장식성 텍스트는 절대 넣지 말 것

[콘텐츠 주입 데이터 (3x3)]
- 회사명: {company_name}
- BM 유형: {clean_korean_label(bmc_data.get("bm_type", ""), fallback="플랫폼형")}
- [상단-좌측] 시장/니즈: {market_needs}
- [상단-중앙] 타겟 고객: {target_users}
- [상단-우측] 채널/소통: {community_channels}
- [중단-좌측] 제공 가치: {value_prop}
- [중단-중앙] 코어 플랫폼: {core_platform}
- [중단-우측] 핵심 활동: {activities}
- [하단-좌측] 핵심 파트너: {partners}
- [하단-중앙] 기업 본체: {company}
- [하단-우측] 핵심 자원: {resources}

[흐름 라벨]
- 정보 흐름: {info_flows}
- 돈 흐름: {money_flows}
- 서비스 흐름: {service_flows}

[검증된 화살표 방향]
{validated_flow_lines}

[중요 제약]
- 카드끼리 겹침 금지
- 화살표와 라벨이 카드나 글자를 관통하지 않도록 곡선형으로 우회할 것
- polished consulting-style ecosystem diagram 으로 보이게 할 것
- 범례 누락 금지
- 화살표 방향은 반드시 위의 [검증된 화살표 방향]을 그대로 따를 것
- 위의 방향 목록에 없는 추가 화살표는 만들지 말 것
- 각 화살표는 반드시 시작 주체 카드 경계에서 출발해 도착 주체 카드 경계에서 끝나야 함
- 최종 결과는 PNG 이미지로 생성
""".strip()


def _join_items(values: List[Any], fallback: str) -> str:
    items = []
    for value in values or []:
        cleaned = clean_korean_label(value)
        if cleaned and cleaned not in items:
            items.append(cleaned)
        if len(items) >= 3:
            break
    return ", ".join(items) if items else fallback


def _join_flow_labels(flows: List[Dict[str, str]], fallback: str) -> str:
    labels = []
    for flow in flows or []:
        label = clean_korean_label(flow.get("label", ""))
        if label and label not in labels:
            labels.append(label)
        if len(labels) >= 4:
            break
    return ", ".join(labels) if labels else fallback


def _validate_role_flows(
    client: genai.Client,
    company_name: str,
    bmc_data: Dict[str, Any],
) -> List[Dict[str, str]]:
    draft_flows, ambiguous_flows = _build_rule_based_role_flows(bmc_data)
    if ambiguous_flows:
        repaired = _repair_ambiguous_flows_with_model(client, company_name, bmc_data, draft_flows, ambiguous_flows)
        if repaired:
            return repaired[:8]
    return draft_flows[:8]


def _build_rule_based_role_flows(bmc_data: Dict[str, Any]) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    flows: List[Dict[str, str]] = []
    ambiguous: List[Dict[str, str]] = []

    for flow_type, source_key in [
        ("정보", "information_flows"),
        ("돈", "money_flows"),
        ("서비스", "service_flows"),
    ]:
        for item in bmc_data.get(source_key, [])[:4]:
            label = clean_korean_label(item.get("label", ""))
            if not label:
                continue
            inferred = _infer_role_flow(flow_type, label, bmc_data)
            if inferred:
                flows.append({"type": flow_type, **inferred})
            else:
                ambiguous.append({"type": flow_type, "label": label})

    flows = _dedupe_role_flows(flows)
    return flows[:8], ambiguous[:6]


def _infer_role_flow(flow_type: str, label: str, bmc_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    bmc = bmc_data.get("business_model_canvas", {}) or {}
    revenue_text = " ".join(bmc.get("revenue_streams", []))
    cost_text = " ".join(bmc.get("cost_structure", []))
    channel_text = " ".join(bmc.get("channels", []))
    relation_text = " ".join(bmc.get("customer_relationships", []))
    resource_text = " ".join(bmc.get("key_resources", []))
    activity_text = " ".join(bmc.get("key_activities", []))

    if flow_type == "정보":
        if any(token in label for token in ["사용", "요청", "문의", "입력", "행동"]):
            return {"from": "타겟 고객", "to": "코어 플랫폼", "label": label}
        if any(token in label for token in ["도입", "리드", "채널", "영업"]):
            return {"from": "커뮤니티 및 채널", "to": "코어 플랫폼", "label": label}
        if any(token in label for token in ["모델", "원천", "기술", "데이터", "API", "연동"]) or any(
            token in label for token in ["모델", "원천", "기술", "데이터"]
        ):
            return {"from": "핵심 자원", "to": "코어 플랫폼", "label": label}
        return None

    if flow_type == "돈":
        if any(token in label for token in ["인프라", "클라우드", "호스팅", "서버"]):
            return {"from": "기업 본체", "to": "핵심 자원", "label": label}
        if any(token in label for token in ["모델", "데이터", "라이선스"]):
            return {"from": "기업 본체", "to": "핵심 파트너", "label": label}
        if any(token in label for token in ["제휴", "리셀", "채널", "도입", "파트너"]) and any(
            token in revenue_text + channel_text + relation_text for token in ["수수료", "제휴", "리셀", "도입", "파트너"]
        ):
            return {"from": "커뮤니티 및 채널", "to": "기업 본체", "label": label}
        if any(token in label for token in ["구독", "이용", "사용", "멤버십", "가입"]) or any(
            token in revenue_text for token in ["구독", "이용", "멤버십", "가입", "사용료"]
        ):
            return {"from": "타겟 고객", "to": "기업 본체", "label": label}
        if "수수료" in label and "파트너" not in revenue_text + channel_text + relation_text:
            return {"from": "타겟 고객", "to": "기업 본체", "label": label}
        if any(token in cost_text for token in ["인프라", "클라우드"]) and "비용" in label:
            return {"from": "기업 본체", "to": "핵심 자원", "label": label}
        if any(token in cost_text for token in ["모델", "데이터", "라이선스"]) and "비용" in label:
            return {"from": "기업 본체", "to": "핵심 파트너", "label": label}
        return None

    if flow_type == "서비스":
        if any(token in label for token in ["인프라", "클라우드", "호스팅", "서버"]):
            return {"from": "핵심 자원", "to": "코어 플랫폼", "label": label}
        if any(token in label for token in ["솔루션", "도입", "채널", "영업", "제휴"]):
            return {"from": "코어 플랫폼", "to": "커뮤니티 및 채널", "label": label}
        if any(token in label for token in ["API", "연동"]) and "도입" not in label:
            return {"from": "코어 플랫폼", "to": "핵심 활동", "label": label}
        if any(token in label for token in ["기술", "모델", "데이터", "원천"]):
            return {"from": "핵심 파트너", "to": "코어 플랫폼", "label": label}
        if any(token in label for token in ["보안", "분석", "탐지", "결과", "추천", "대응", "서비스"]) and not any(
            token in label for token in ["인프라", "연동", "API", "도입", "제휴"]
        ):
            return {"from": "코어 플랫폼", "to": "타겟 고객", "label": label}
        return None

    return None


def _repair_ambiguous_flows_with_model(
    client: genai.Client,
    company_name: str,
    bmc_data: Dict[str, Any],
    draft_flows: List[Dict[str, str]],
    ambiguous_flows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    schema_hint = """
{
  "validated_role_flows": [
    {"type": "정보", "from": "Users", "to": "Core Platform", "label": ""}
  ]
}
""".strip()
    prompt = f"""
당신은 비즈니스 생태계 다이어그램 검증기이다.

{company_name}의 기업 정보와 BMC를 바탕으로, 아래 흐름들의 방향이 맞는지 판단하고
최종 role-based flow 목록만 JSON으로 출력하라.

[허용 role]
- 시장 상황 및 니즈
- 타겟 고객
- 커뮤니티 및 채널
- 제공 가치
- 코어 플랫폼
- 핵심 활동
- 핵심 파트너
- 기업 본체
- 핵심 자원

[판단 우선순위]
1. business_model_canvas
2. bmc_summary / strategic_summary
3. flow label

[기존 확정 흐름]
{json.dumps(draft_flows, ensure_ascii=False, indent=2)}

[애매한 흐름]
{json.dumps(ambiguous_flows, ensure_ascii=False, indent=2)}

[BMC 데이터]
{json.dumps(bmc_data, ensure_ascii=False, indent=2)}

[규칙]
- 강제 고정이 아니라 맥락 기반으로 판단
- 방향이 틀렸다면 수정
- 불필요한 흐름은 제거 가능
- 최종 흐름은 최대 8개
- JSON ONLY

[출력 스키마]
{schema_hint}
"""
    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="text/plain"),
    )
    raw_text = (response.text or "").strip()
    try:
        data = extract_json(raw_text)
    except Exception:
        repaired = repair_json_with_model(client, TEXT_MODEL, raw_text, schema_hint=schema_hint)
        data = extract_json(repaired)

    flows = []
    for item in data.get("validated_role_flows", []) or []:
        flow_type = clean_korean_label(item.get("type", ""))
        start = clean_korean_label(item.get("from", ""))
        end = clean_korean_label(item.get("to", ""))
        label = clean_korean_label(item.get("label", ""))
        if flow_type in {"정보", "돈", "서비스"} and start and end and label:
            flows.append({"type": flow_type, "from": start, "to": end, "label": label})
    return _dedupe_role_flows(flows)


def _dedupe_role_flows(flows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    deduped = []
    for flow in flows:
        key = (flow["type"], flow["from"], flow["to"], flow["label"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(flow)
    return deduped


def _format_validated_flows(flows: List[Dict[str, str]]) -> str:
    if not flows:
        return "- 정보: 타겟 고객 -> 코어 플랫폼 : 사용 데이터"
    return "\n".join(f'- {flow["type"]}: {flow["from"]} -> {flow["to"]} : {flow["label"]}' for flow in flows[:8])


def _extract_flow_labels(flows: List[Dict[str, str]]) -> List[str]:
    labels = []
    for flow in flows or []:
        label = clean_korean_label(flow.get("label", ""))
        if label and label not in labels:
            labels.append(label)
        if len(labels) >= 4:
            break
    return labels
