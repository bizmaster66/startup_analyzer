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
- 투자자 및 컨설팅 슬라이드에 즉시 넣을 수 있는 전문가 수준의 비즈니스 작동 원리 도해도를 생성할 것
- 복잡한 장식이나 실험적인 플로우차트를 절대 금지하며, 깔끔하고 정돈된 ecosystem mechanism diagram 형태를 유지할 것
- 전체 톤앤매너는 미니멀한 컨설팅 스타일의 ecosystem map처럼 보이게 할 것
- 현재 구성과 콘텐츠는 유지하되, 시각적 인상은 카드형 BMC 요약이 아니라 '이 기업이 누구에게 무엇을 어떻게 전달하고 어디서 돈과 정보가 흐르는지'가 한눈에 보이는 전략 다이어그램이어야 함
- 이 이미지를 보면 해당 기업의 비즈니스가 어떻게 작동하는지 즉시 이해되어야 함

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
- 폰트/텍스트: 전체 텍스트는 지금 결과물의 체감상 절반 수준까지 축소할 것
- 폰트/텍스트: title과 bullet의 최대 글자 크기를 엄격히 제한해 카드 내부를 지배하지 않게 할 것
- 폰트/텍스트: title은 node label 수준의 작은 전문 제목으로 처리하고, headline처럼 크게 키우지 말 것
- 폰트/텍스트: subtitle은 모든 노드에서 금지한다. title 아래에 subtitle 줄을 추가하지 말 것
- 폰트/텍스트: bullet은 title보다 더 작은 크기로 처리하고, 1줄당 짧은 명사구/실무 키워드만 넣을 것
- 폰트/텍스트: title 최대 크기는 카드 높이의 약 12%를 넘기지 말 것
- 폰트/텍스트: bullet 최대 크기는 title의 65% 수준을 넘기지 말 것
- 폰트/텍스트: 텍스트 블록 전체가 카드 높이의 35%를 넘지 않게 하고, 카드 안에 빈 여백이 충분히 남아야 함
- 폰트/텍스트: 글자 두께, 줄간격, 여백이 전문적인 corporate slide 품질로 보이게 할 것
- 폰트/텍스트: 타이틀과 bullet의 정렬 기준이 일관되게 유지되도록 할 것
- 폰트/텍스트: 긴 문장을 강제로 줄바꿈하지 말고, 더 짧은 표현으로 축약해 가독성을 확보할 것
- 폰트/텍스트: 샘플 전략 다이어그램처럼 title은 semibold 또는 bold, bullet은 regular 또는 medium으로 처리할 것
- 폰트/텍스트: 텍스트 색은 검정 또는 매우 짙은 회색으로 유지하고, 본문에 불필요한 색상 강조를 넣지 말 것
- 각 노드는 title 1줄, bullet 1~2개만 포함
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
- 상단 왼쪽(1-1) 타이틀은 반드시 "Problem"으로 표기하고, 시장의 핵심 문제와 미충족 수요를 설명할 것
- 상단 중앙(1-2) 타이틀은 반드시 "Target"으로 표기하고, 핵심 고객/사용자를 설명할 것
- 상단 오른쪽(1-3) 타이틀은 반드시 "Channel"로 표기하고, 기업과 고객의 접점 및 영업 방식을 설명할 것
- 중앙 왼쪽(2-1) 타이틀은 반드시 "Partner"로 표기하고, 기업과의 관계를 설명할 것
- 중앙 중앙(2-2)은 고정 타이틀을 쓰지 말고, 기업의 핵심 사업을 가장 잘 나타내는 키워드를 title로 사용할 것
- 중앙 중앙(2-2)은 자산, 플랫폼, 서비스, 기술 중 핵심 비즈니스 실체를 가장 잘 드러내야 하며, 시각적으로 중심 노드로 강조할 것
- 중앙 오른쪽(2-3) 타이틀은 반드시 "Operating"으로 표기하고, 핵심 사업을 수행하기 위한 운영 활동을 설명할 것
- 하단 왼쪽(3-1) 타이틀은 반드시 "Value Proposition"으로 표기하고, 고객이 경쟁사 대신 이 기업을 선택해야 하는 차별적 효익을 설명할 것
- 하단 중앙(3-2)은 "기업 본체"라는 표현을 절대 쓰지 말고, title에 반드시 실제 기업명 {company_name} 을 사용할 것
- 하단 중앙(3-2)은 수익의 주체이자 비용 집행의 주체로 표현할 것
- 하단 오른쪽(3-3) 타이틀은 반드시 "Moat"으로 표기하고, 핵심 자원과 경쟁력을 설명할 것
- 별도 메인 타이틀이나 장식성 텍스트는 절대 넣지 말 것

[콘텐츠 주입 데이터 (3x3)]
- 회사명: {company_name}
- BM 유형: {clean_korean_label(bmc_data.get("bm_type", ""), fallback="플랫폼형")}
- [상단-좌측 Problem] 시장 문제: {market_needs}
- [상단-중앙 Target] 핵심 고객: {target_users}
- [상단-우측 Channel] 고객 접점/영업: {community_channels}
- [중단-좌측 Partner] 파트너 관계: {partners}
- [중단-중앙 Core] 핵심 사업 키워드: {core_platform}
- [중단-우측 Operating] 핵심 활동: {activities}
- [하단-좌측 Value Proposition] 차별적 효익: {value_prop}
- [하단-중앙 Company] 실제 기업명: {company}
- [하단-우측 Moat] 핵심 자원/경쟁력: {resources}

[노드 텍스트 작성 방식]
- 3x3 각 칸의 title은 위에서 지정한 영문 title을 반드시 그대로 사용할 것. 단, 중앙 중앙(2-2)만 예외적으로 핵심 사업 키워드를 title로 사용
- 하단 중앙(3-2)은 title로 반드시 실제 기업명만 표기하고, "기업 본체" 같은 일반 명칭은 금지
- 각 노드의 title은 명사형 professional business label로 작성할 것
- 각 노드의 title은 추상적인 수식어나 광고성 표현 없이 역할이 바로 드러나게 작성할 것
- subtitle은 모든 노드에서 사용 금지
- 각 노드의 bullet은 1~2개만 사용하고, 모두 짧은 실무 키워드 또는 명사구로 작성할 것
- 각 노드의 bullet은 동사형 문장보다 제품 기능, 운영 요소, 고객 특성, 파트너 기능처럼 업무적으로 읽히는 키워드 중심으로 작성할 것
- 각 노드의 bullet은 한 항목당 10자 내외의 짧은 표현을 우선하고, 장문 서술형 문장은 금지할 것
- Problem은 시장 문제, Target은 고객, Channel은 접점/영업, Partner는 관계, Operating은 운영 활동, Value Proposition은 차별 효익, Moat은 경쟁 우위를 드러내는 내용만 써야 함
- 중앙 중앙(2-2)은 핵심 사업 자체를 나타내는 짧은 키워드를 title로 쓰고, bullet로만 사업 실체를 설명할 것
- 하단 중앙(3-2)은 돈 흐름이 들어오고 비용이 나가는 재무 주체로 이해되도록 표현할 것
- 각 노드의 텍스트는 투자자 보고서, 전략 컨설팅 산출물, 사업개발 제안서에 들어갈 만한 톤으로 정제할 것
- 캐주얼한 표현, 홍보성 표현, 문장형 장문 bullet 금지
- 노드 텍스트는 투자자 보고서에 들어가는 용어 수준으로 정제할 것

[흐름 중심 구성 원칙]
- 이 이미지는 BMC 카드 요약이 아니라 비즈니스가 어떻게 작동하는지 보여주는 도해도여야 함
- 노드 내부 텍스트보다 화살표와 화살표 라벨이 더 먼저 읽히도록 구성할 것
- 화살표는 충분히 길고 선명하게 보여야 하며, 카드 사이 여백을 넓게 확보해 화살표가 숨지 않게 할 것
- 각 주요 관계마다 화살표 라벨을 붙여, 무엇이 이동하는지 즉시 이해되게 할 것
- 화살표 라벨은 "사용 데이터", "도입 요청", "분석 결과", "구독료", "제휴 수수료", "채널 지원", "운영 데이터", "인프라 비용"처럼 이동 대상이 드러나는 명확한 워딩을 사용할 것
- 흐름 라벨은 추상어보다 교환되는 정보, 돈, 서비스/자산을 직접 설명하는 표현을 우선할 것
- 화살표 수는 너무 적지 않게 유지하되, 핵심 메커니즘을 보여주는 주요 흐름이 충분히 보이게 할 것
- 중앙의 핵심 사업 노드와 하단 중앙의 {company_name} 사이의 역할 차이가 화살표로 분명히 드러나야 함

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
- 노드 설명이 너무 많아서 BMC 이미지처럼 보이면 실패다
- 화살표가 적거나 짧아서 작동 구조가 안 보이면 실패다
- 최종 이미지는 '무엇을 팔고, 누구와 연결되며, 돈과 정보와 서비스가 어떻게 움직이는지'를 이해할 수 있어야 한다
- 범례 누락 금지
- 화살표 방향은 반드시 위의 [검증된 화살표 방향]을 그대로 따를 것
- 위의 방향 목록에 없는 추가 화살표는 만들지 말 것
- 각 화살표는 반드시 시작 주체 카드 경계에서 출발해 도착 주체 카드 경계에서 끝나야 함
- 돈, 정보, 서비스/자산 화살표는 각각 실제 주고받는 관계에 맞게 논리적으로 연결할 것
- 하단 중앙의 {company_name} 은 수익의 수취 주체와 비용 지출 주체가 되어야 하며, 이 역할이 흐름에서 드러나야 함
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
