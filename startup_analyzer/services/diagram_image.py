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
- 전체 톤앤매너는 미니멀한 컨설팅 스타일의 ecosystem map처럼 보이게 할 것
- 현재 구성과 콘텐츠는 유지하되, 시각적 인상은 가벼운 카드형 대시보드보다 정제된 전략 다이어그램에 가깝게 만들 것

[시각 스타일]
- 배경: 흰색 또는 아주 연한 회색
- 카드: 필요할 경우에만 아주 연한 회색 또는 흰색 배경의 미세한 rounded rectangle을 사용하고, 두꺼운 카드 박스나 강한 그림자는 금지
- 카드: 카드보다 노드 라벨과 흐름이 먼저 보이도록, 프레임은 매우 얇고 절제된 outline 스타일로 처리
- 테마: 흰 배경, 짙은 회색 또는 검은 텍스트, 정보/돈/서비스 흐름만 색을 사용하는 컨설팅 다이어그램 톤
- 아이콘: 이모지나 캐주얼한 이모티콘 사용 금지
- 아이콘: 각 노드에는 발표자료/컨설팅 슬라이드에 적합한 전문적인 monochrome line icon 또는 simple flat business icon만 작게 배치
- 아이콘: 검정 또는 짙은 회색 중심의 절제된 corporate UI icon style을 사용할 것
- 아이콘: 귀엽거나 장식적인 그림체, 스티커 느낌, 3D 느낌, colorful emoji style 금지
- 폰트/텍스트: 전문 보고서용 슬라이드처럼 가독성이 높은 sans-serif 스타일로 표현
- 폰트/텍스트: 전체 텍스트는 현재 기본보다 확실히 축소하되, 발표자료에서 읽기 어려울 정도로 작게 만들지는 말 것
- 폰트/텍스트: title, subtitle, bullet 각각의 최대 글자 크기를 엄격히 제한해 카드 내부를 지배하지 않게 할 것
- 폰트/텍스트: title은 node label 수준의 작은 전문 제목으로 처리하고, headline처럼 크게 키우지 말 것
- 폰트/텍스트: subtitle은 title보다 한 단계 작은 보조 텍스트로 처리하고, 설명문이 아니라 짧은 business descriptor만 넣을 것
- 폰트/텍스트: bullet은 subtitle보다 더 작은 크기로 처리하고, 1줄당 짧은 명사구/실무 키워드만 넣을 것
- 폰트/텍스트: title 최대 크기는 카드 높이의 약 12%를 넘기지 말 것
- 폰트/텍스트: subtitle 최대 크기는 title의 80% 수준을 넘기지 말 것
- 폰트/텍스트: bullet 최대 크기는 title의 65% 수준을 넘기지 말 것
- 폰트/텍스트: 텍스트 블록 전체가 카드 높이의 55%를 넘지 않게 하고, 카드 안에 빈 여백이 충분히 남아야 함
- 폰트/텍스트: 글자 두께, 줄간격, 여백이 전문적인 corporate slide 품질로 보이게 할 것
- 폰트/텍스트: 타이틀, 서브타이틀, bullet 모두 중앙 또는 좌측 정렬 기준이 일관되게 유지되도록 할 것
- 폰트/텍스트: 긴 문장을 강제로 줄바꿈하지 말고, 더 짧은 표현으로 축약해 가독성을 확보할 것
- 폰트/텍스트: 샘플 전략 다이어그램처럼 title은 semibold 또는 bold, subtitle과 bullet은 regular 또는 medium으로 처리할 것
- 폰트/텍스트: 텍스트 색은 검정 또는 매우 짙은 회색으로 유지하고, 본문에 불필요한 색상 강조를 넣지 말 것
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

[노드 텍스트 작성 방식]
- 각 노드의 title은 명사형 professional business label로 작성할 것
- 각 노드의 title은 추상적인 수식어나 광고성 표현 없이 역할이 바로 드러나게 작성할 것
- 각 노드의 subtitle은 한 줄짜리 역할/기능 요약으로 작성하고, 완전한 문장형 설명은 금지할 것
- 각 노드의 bullet은 2~3개만 사용하고, 모두 짧은 실무 키워드 또는 명사구로 작성할 것
- 각 노드의 bullet은 동사형 문장보다 제품 기능, 운영 요소, 고객 특성, 파트너 기능처럼 업무적으로 읽히는 키워드 중심으로 작성할 것
- 각 노드의 bullet은 한 항목당 12자 내외의 짧은 표현을 우선하고, 장문 서술형 문장은 금지할 것
- 각 노드의 텍스트는 투자자 보고서, 전략 컨설팅 산출물, 사업개발 제안서에 들어갈 만한 톤으로 정제할 것
- 캐주얼한 표현, 홍보성 표현, 문장형 장문 bullet 금지
- 노드 텍스트는 투자자 보고서에 들어가는 용어 수준으로 정제할 것

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
- 샘플 전략 다이어그램처럼 넓은 여백, 절제된 아이콘, 작은 텍스트, 선 중심의 구조를 유지할 것
- 과한 카드 장식, 큰 제목, 큰 아이콘, 만화 같은 스타일, 이모지 스타일을 절대 사용하지 말 것
- 범례 누락 금지
- 화살표 방향은 반드시 위의 [검증된 화살표 방향]을 그대로 따를 것
- 위의 방향 목록에 없는 추가 화살표는 만들지 말 것
- 각 화살표는 반드시 시작 주체 카드 경계에서 출발해 도착 주체 카드 경계에서 끝나야 함
- 전체 톤앤매너와 룩앤필은 캐주얼하지 않고, 전문 컨설팅 보고서/투자자 자료 수준으로 유지할 것
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
