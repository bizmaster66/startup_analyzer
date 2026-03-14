from typing import Any, Dict, List

from google import genai

from startup_analyzer.utils.text import clean_korean_label


IMAGE_MODEL_NAME = "gemini-2.5-flash-image"


def generate_bm_diagram_png(
    client: genai.Client,
    company_name: str,
    bmc_data: Dict[str, Any],
) -> bytes:
    prompt = _build_diagram_prompt(company_name, bmc_data)
    response = client.models.generate_content(
        model=IMAGE_MODEL_NAME,
        contents=[prompt],
    )

    for part in getattr(response, "parts", []) or []:
        inline_data = getattr(part, "inline_data", None)
        if inline_data and getattr(inline_data, "data", None):
            return inline_data.data

    raise ValueError("Gemini 이미지 생성 응답에서 PNG 데이터를 찾지 못했습니다.")


def _build_diagram_prompt(company_name: str, bmc_data: Dict[str, Any]) -> str:
    bmc = bmc_data.get("business_model_canvas", {}) or {}

    users = _join_items(bmc.get("customer_segments", []), fallback="핵심 고객")
    providers = _join_items(bmc.get("key_resources", []), fallback="데이터·기술 공급")
    partners = _join_items(bmc.get("channels", []), fallback="도입 파트너")
    consumers = _join_items(bmc.get("key_activities", []), fallback="서비스 활용 주체")
    infrastructure = _join_items(bmc.get("cost_structure", []), fallback="인프라 파트너")
    platform = clean_korean_label(bmc_data.get("middle_layer", ""), fallback=f"{company_name} 플랫폼")

    info_flows = _join_flow_labels(bmc_data.get("information_flows", []), fallback="사용 데이터, 도입 정보")
    money_flows = _join_flow_labels(bmc_data.get("money_flows", []), fallback="구독 매출, 제휴 수수료")
    service_flows = _join_flow_labels(bmc_data.get("service_flows", []), fallback="분석 결과, 인프라 제공")

    return f"""
한국어 비즈니스 생태계 다이어그램 PNG를 생성하라.

[목표]
- 컨설팅 슬라이드에 넣을 수준의 깔끔한 business ecosystem diagram
- 흰색 또는 아주 연한 회색 배경
- 중심 플랫폼 1개와 주변 주체 5개가 균형 있게 배치된 구조
- 복잡한 장식 금지, 과한 일러스트 금지

[레이아웃]
- 상단: Users
- 중앙: Core Platform
- 좌측: Providers
- 우측: Partners
- 좌하단: Consumers
- 우하단: Infrastructure
- 중앙 플랫폼을 기준으로 대칭적이고 안정된 구도

[시각 스타일]
- rounded rectangle cards
- 옅은 블루 카드 배경
- 진한 네이비 텍스트
- 곡선형 연결선
- 정보 흐름은 파란색 + 사각형 마커
- 돈 흐름은 초록색 + 달러 마커
- 서비스 흐름은 주황색 + 원형 마커
- 각 노드는 title, subtitle, bullet 2~3개만 포함
- 글자는 모두 한국어
- 전체 화면 비율은 가로형 16:10에 가깝게

[콘텐츠]
- 회사명: {company_name}
- BM 유형: {clean_korean_label(bmc_data.get("bm_type", ""), fallback="플랫폼형")}
- Users: {users}
- Core Platform: {platform}
- Providers: {providers}
- Partners: {partners}
- Consumers: {consumers}
- Infrastructure: {infrastructure}

[핵심 가치]
- { _join_items(bmc.get("value_propositions", []), fallback="핵심 가치") }

[흐름 라벨]
- 정보 흐름: {info_flows}
- 돈 흐름: {money_flows}
- 서비스 흐름: {service_flows}

[중요 제약]
- 텍스트 과밀 금지
- 카드끼리 겹침 금지
- 화살표와 라벨 충돌 금지
- 실험적인 플로우차트처럼 보이면 안 됨
- polished consulting-style ecosystem diagram 으로 보이게 할 것
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
