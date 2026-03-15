import json
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from startup_analyzer.services.analysis import TEXT_MODEL
from startup_analyzer.utils.json_utils import extract_json, repair_json_with_model
from startup_analyzer.utils.text import clean_korean_label


IMAGE_MODEL = "gemini-3.1-flash-image-preview"
GENERIC_PARTNER_TERMS = {"전략적 투자 기관", "정부 R&D 기관", "정부 연구 기관", "투자 기관", "R&D 기관"}
PARTNER_EXCLUDE_KEYWORDS = ["투자", "인베스트", "펀드", "VC", "액셀러레이터", "국책과제"]


def generate_bm_diagram_png(
    client: genai.Client,
    company_name: str,
    bmc_data: Dict[str, Any],
) -> bytes:
    validated_flows = _validate_role_flows(client, company_name, bmc_data)
    node_specs = _prepare_node_specs(client, company_name, bmc_data)
    prompt = _build_diagram_prompt(company_name, bmc_data, validated_flows, node_specs)
    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=[prompt],
    )

    for part in getattr(response, "parts", []) or []:
        inline_data = getattr(part, "inline_data", None)
        if inline_data and getattr(inline_data, "data", None):
            return inline_data.data

    raise ValueError("Gemini 이미지 생성 응답에서 PNG 데이터를 찾지 못했습니다.")


def _build_diagram_prompt(
    company_name: str,
    bmc_data: Dict[str, Any],
    validated_flows: List[Dict[str, str]],
    node_specs: Dict[str, Dict[str, Any]],
) -> str:
    bmc = bmc_data.get("business_model_canvas", {}) or {}

    info_flows = _join_validated_labels_by_type(validated_flows, "정보", fallback="사용 데이터, 요청 정보")
    money_flows = _join_validated_labels_by_type(validated_flows, "돈", fallback="구독료, 수수료")
    service_flows = _join_validated_labels_by_type(validated_flows, "서비스", fallback="핵심 서비스, 플랫폼 운영")
    validated_flow_lines = _format_validated_flows(validated_flows)
    node_spec_lines = _format_node_specs(node_specs)

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
- 3x3 좌표 번호(예: 1-1, 2-3) 같은 템플릿 번호 표시는 절대 넣지 말 것

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
- 아래 node spec을 그대로 사용하고, generic placeholder 문구로 바꾸지 말 것
{node_spec_lines}

[노드 텍스트 작성 방식]
- 3x3 각 칸의 title은 위에서 지정한 영문 title을 반드시 그대로 사용할 것. 단, 중앙 중앙(2-2)만 예외적으로 핵심 사업 키워드를 title로 사용
- 하단 중앙(3-2)은 title로 반드시 실제 기업명만 표기하고, "기업 본체" 같은 일반 명칭은 금지
- 각 노드의 title은 명사형 professional business label로 작성할 것
- 각 노드의 title은 추상적인 수식어나 광고성 표현 없이 역할이 바로 드러나게 작성할 것
- subtitle은 모든 노드에서 사용 금지
- 각 노드의 bullet은 1~2개만 사용하고, 모두 짧은 실무 키워드 또는 명사구로 작성할 것
- 각 노드의 bullet은 동사형 문장보다 제품 기능, 운영 요소, 고객 특성, 파트너 기능처럼 업무적으로 읽히는 키워드 중심으로 작성할 것
- 각 노드의 bullet은 한 항목당 10자 내외의 짧은 표현을 우선하고, 장문 서술형 문장은 금지할 것
- 각 노드는 위 [콘텐츠 주입 데이터]에서 준 bullet만 사용하고, 임의로 장문 bullet을 추가하지 말 것
- "시장의 핵심 문제와 미충족 수요", "핵심 고객 및 사용자", "기업과 고객의 접점 및 영업 방식", "기업과의 관계", "핵심 사업 수행을 위한 운영 활동", "핵심 자원과 경쟁력" 같은 템플릿 문구를 bullet로 다시 쓰지 말 것
- "core business keyword", "platform business", "기업 본체" 같은 메타 표현이나 placeholder 표현을 절대 출력하지 말 것
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
- 노드 안에 bullet이 3개 이상 들어가면 실패다
- title 아래 설명형 두 번째 헤더나 subtitle이 나오면 실패다
- node spec의 실제 내용을 무시하고 generic bullet로 대체하면 실패다
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


def _prepare_node_specs(client: genai.Client, company_name: str, bmc_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    specs = _build_default_node_specs(company_name, bmc_data)
    if _needs_node_spec_repair(specs):
        repaired = _repair_node_specs_with_model(client, company_name, bmc_data, specs)
        if repaired:
            specs = repaired
    return specs


def _build_default_node_specs(company_name: str, bmc_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    bmc = bmc_data.get("business_model_canvas", {}) or {}
    archetype = _infer_business_archetype(bmc_data)
    return {
        "problem": {"title": "Problem", "bullets": _problem_items(bmc_data, archetype)},
        "target": {"title": "Target", "bullets": _target_items(bmc_data, archetype)},
        "channel": {"title": "Channel", "bullets": _channel_items(bmc_data, archetype)},
        "partner": {"title": "Partner", "bullets": _partner_items(bmc_data, archetype)},
        "core": {
            "title": clean_korean_label(bmc_data.get("middle_layer", ""), fallback=f"{company_name} 플랫폼"),
            "bullets": _core_items(bmc_data, archetype),
        },
        "operating": {"title": "Operating", "bullets": _operating_items(bmc_data, archetype)},
        "value": {"title": "Value Proposition", "bullets": _value_items(bmc_data, archetype)},
        "company": {"title": company_name, "bullets": _company_items(bmc_data, company_name)},
        "moat": {"title": "Moat", "bullets": _moat_items(bmc_data, archetype)},
    }


def _infer_business_archetype(bmc_data: Dict[str, Any]) -> str:
    bmc = bmc_data.get("business_model_canvas", {}) or {}
    combined = " ".join(
        [
            clean_korean_label(bmc_data.get("bm_type", "")),
            clean_korean_label(bmc_data.get("middle_layer", "")),
            " ".join(bmc.get("customer_segments", [])),
            " ".join(bmc.get("value_propositions", [])),
            " ".join(bmc.get("channels", [])),
            " ".join(bmc.get("key_resources", [])),
            " ".join(bmc.get("key_activities", [])),
            " ".join(bmc.get("key_partnerships", [])),
        ]
    )
    if any(token in combined for token in ["뷰티", "브랜드", "색조", "네일", "립", "걸코어", "화장품"]):
        return "brand_consumer"
    if any(token in combined for token in ["로봇", "자율주행", "RBS", "바리스타", "배달", "휴머노이드"]):
        return "robotics_b2b"
    if any(token in combined for token in ["커머스", "앱테크", "마켓", "쇼핑", "플랫폼", "소비자"]):
        return "commerce_platform"
    return "generic"


def _core_items(bmc_data: Dict[str, Any], archetype: str) -> List[str]:
    bmc = bmc_data.get("business_model_canvas", {}) or {}
    candidates: List[str] = []
    for value in [bmc_data.get("middle_layer", "")]:
        phrase = _core_phrase(value, archetype)
        if phrase and phrase not in candidates:
            candidates.append(phrase)
        if len(candidates) >= 1:
            break
    for value in bmc.get("key_activities", []):
        phrase = _core_phrase(value, archetype)
        if phrase and phrase not in candidates:
            candidates.append(phrase)
        if len(candidates) >= 2:
            break
    items: List[str] = []
    for value in candidates:
        cleaned = _short_phrase(value, max_len=12)
        if cleaned and cleaned not in items:
            items.append(cleaned)
        if len(items) >= 2:
            break
    return items or ["핵심 서비스"]


def _target_items(bmc_data: Dict[str, Any], archetype: str) -> List[str]:
    bmc = bmc_data.get("business_model_canvas", {}) or {}
    items: List[str] = []
    for value in bmc.get("customer_segments", []):
        text = clean_korean_label(value)
        if not text:
            continue
        phrase = ""
        if archetype == "brand_consumer" and "MZ세대" in text:
            phrase = "MZ세대 여성"
        elif archetype == "brand_consumer" and "뷰티" in text and "소비자" in text:
            phrase = "뷰티 소비자"
        elif archetype == "brand_consumer" and "프리미엄" in text and "소비자" in text:
            phrase = "프리미엄 색조 소비자"
        elif archetype == "robotics_b2b" and any(token in text for token in ["운영사", "리테일", "시설", "관리자"]):
            phrase = _short_phrase(text, max_len=14)
        else:
            phrase = _short_phrase(text, max_len=14)
        if phrase and phrase not in items:
            items.append(phrase)
        if len(items) >= 2:
            break
    return items or ["핵심 고객"]


def _channel_items(bmc_data: Dict[str, Any], archetype: str) -> List[str]:
    bmc = bmc_data.get("business_model_canvas", {}) or {}
    items: List[str] = []
    for value in bmc.get("channels", []):
        text = clean_korean_label(value)
        if not text:
            continue
        phrase = ""
        if archetype == "robotics_b2b" and "직영" in text and "카페" in text:
            phrase = "직영 로봇 카페"
        elif archetype == "brand_consumer" and ("올리브영" in text or "H&B" in text):
            phrase = "올리브영 H&B"
        elif archetype == "brand_consumer" and "온라인" in text and "스토어" in text:
            phrase = "공식 온라인 스토어"
        elif archetype == "brand_consumer" and ("SNS" in text or "인플루언서" in text):
            phrase = "SNS·인플루언서"
        elif "직접 영업" in text or "기업 고객" in text:
            phrase = "기업 직접 영업"
        elif "전시" in text:
            phrase = "전시회 리드"
        elif "앱" in text or "온라인" in text:
            phrase = _short_phrase(text, max_len=14)
        else:
            phrase = _short_phrase(text, max_len=14)
        if phrase and phrase not in items:
            items.append(phrase)
        if len(items) >= 2:
            break
    return items or ["고객 접점"]


def _partner_items(bmc_data: Dict[str, Any], archetype: str) -> List[str]:
    bmc = bmc_data.get("business_model_canvas", {}) or {}
    items: List[str] = []
    for value in bmc.get("key_partnerships", []):
        text = clean_korean_label(value)
        if not text:
            continue
        if text in GENERIC_PARTNER_TERMS or any(keyword in text for keyword in PARTNER_EXCLUDE_KEYWORDS):
            continue
        if archetype == "brand_consumer" and ("올리브영" in text or "H&B" in text):
            continue
        phrase = ""
        if any(token in text for token in ["OEM", "ODM"]):
            phrase = "OEM·ODM 제조사"
        elif any(token in text for token in ["부품", "하드웨어", "제조"]):
            phrase = "로봇 제조 파트너"
        elif any(token in text for token in ["고객사", "운영사", "매장", "리테일"]):
            phrase = "도입 고객사"
        elif any(token in text for token in ["물류", "배송"]):
            phrase = "물류 운영 파트너"
        else:
            phrase = _short_phrase(text, max_len=14)
        if phrase and phrase not in items:
            items.append(phrase)
        if len(items) >= 2:
            break
    return items or ["제조 파트너", "도입 고객사"]


def _operating_items(bmc_data: Dict[str, Any], archetype: str) -> List[str]:
    bmc = bmc_data.get("business_model_canvas", {}) or {}
    items: List[str] = []
    for value in bmc.get("key_activities", []):
        text = clean_korean_label(value)
        if not text:
            continue
        phrase = ""
        if archetype == "robotics_b2b" and ("R&D" in text or "연구" in text):
            phrase = "로봇 지능 R&D"
        elif archetype == "brand_consumer" and ("OEM" in text or "ODM" in text):
            phrase = "OEM 생산 관리"
        elif "개발" in text and "제조" in text:
            phrase = "시스템 개발·제조"
        elif "유지보수" in text or "설치" in text:
            phrase = "설치·유지보수"
        elif "영업" in text:
            phrase = "B2B 영업"
        elif "매장 운영" in text:
            phrase = "직영 매장 운영"
        else:
            phrase = _short_phrase(text, max_len=14)
        if phrase and phrase not in items:
            items.append(phrase)
        if len(items) >= 2:
            break
    return items or ["핵심 운영"]


def _value_items(bmc_data: Dict[str, Any], archetype: str) -> List[str]:
    bmc = bmc_data.get("business_model_canvas", {}) or {}
    items: List[str] = []
    for value in bmc.get("value_propositions", []):
        phrase = _value_phrase(value, archetype)
        if phrase and phrase not in items:
            items.append(phrase)
        if len(items) >= 2:
            break
    return items or ["차별 효익"]


def _moat_items(bmc_data: Dict[str, Any], archetype: str) -> List[str]:
    bmc = bmc_data.get("business_model_canvas", {}) or {}
    items: List[str] = []
    for value in bmc.get("key_resources", []):
        text = clean_korean_label(value)
        if not text:
            continue
        phrase = ""
        if archetype == "robotics_b2b" and "AI" in text and any(token in text for token in ["지능", "기술", "모델"]):
            phrase = "AI 로봇 지능"
        elif archetype == "brand_consumer" and "브랜드" in text and ("IP" in text or "지식재산" in text):
            phrase = "브랜드 IP"
        elif archetype == "brand_consumer" and (("기획" in text and "디자인" in text) or "미적 경험 디자인" in text):
            phrase = "제품 기획 역량"
        elif archetype == "brand_consumer" and "마케팅" in text and "전문성" in text:
            phrase = "브랜드 마케팅 역량"
        elif "R&D" in text or "엔지니어" in text or "인력" in text:
            phrase = "전문 R&D 인력"
        elif "데이터" in text:
            phrase = "운영 데이터 자산"
        elif "지적 재산" in text or "IP" in text:
            phrase = "로봇 IP"
        else:
            phrase = _short_phrase(text, max_len=14)
        if phrase and phrase not in items:
            items.append(phrase)
        if len(items) >= 2:
            break
    return items or ["AI 로봇 지능", "전문 R&D 인력"]


def _needs_node_spec_repair(specs: Dict[str, Dict[str, Any]]) -> bool:
    generic_bullets = {
        "시장의 핵심 문제와 미충족 수요",
        "핵심 고객 및 사용자",
        "기업과 고객의 접점 및 영업 방식",
        "기업과의 관계",
        "핵심 사업 수행을 위한 운영 활동",
        "핵심 자원과 경쟁력",
    }
    banned_terms = {"core business keyword", "platform business", "기업 본체"}
    for key, spec in specs.items():
        title = clean_korean_label(spec.get("title", ""))
        bullets = [clean_korean_label(x) for x in spec.get("bullets", []) if clean_korean_label(x)]
        if not title or len(bullets) == 0:
            return True
        if len(bullets) > 2:
            return True
        if any(term in title for term in banned_terms):
            return True
        if any(term in bullet for term in banned_terms for bullet in bullets):
            return True
        if any(bullet in generic_bullets for bullet in bullets):
            return True
        if any(bullet in {"합리적", "탐색", "기존", "소비자", "고객", "서비스"} for bullet in bullets):
            return True
        if key == "problem" and any(bullet in {"기존", "소비자", "고객"} for bullet in bullets):
            return True
    return False


def _repair_node_specs_with_model(
    client: genai.Client,
    company_name: str,
    bmc_data: Dict[str, Any],
    specs: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    schema_hint = """
{
  "problem": {"title": "Problem", "bullets": ["", ""]},
  "target": {"title": "Target", "bullets": ["", ""]},
  "channel": {"title": "Channel", "bullets": ["", ""]},
  "partner": {"title": "Partner", "bullets": ["", ""]},
  "core": {"title": "", "bullets": ["", ""]},
  "operating": {"title": "Operating", "bullets": ["", ""]},
  "value": {"title": "Value Proposition", "bullets": ["", ""]},
  "company": {"title": "", "bullets": ["", ""]},
  "moat": {"title": "Moat", "bullets": ["", ""]}
}
""".strip()
    prompt = f"""
당신은 투자자용 비즈니스 다이어그램 편집자이다.

아래 3x3 노드 초안을 더 짧고 구체적인 node spec으로 교정하라.

[목표]
- generic placeholder 문구를 제거
- 각 노드는 bullet 1~2개만 유지
- bullet은 짧은 명사구로 작성
- Problem은 시장 pain, Value Proposition은 차별 효익, Moat은 경쟁우위, Company는 수익/비용 주체를 드러낼 것
- 중앙 core title은 실제 핵심 사업 키워드여야 하며 영어 placeholder 금지

[회사명]
{company_name}

[현재 초안]
{json.dumps(specs, ensure_ascii=False, indent=2)}

[BMC 데이터]
{json.dumps(bmc_data, ensure_ascii=False, indent=2)}

[금지]
- "시장의 핵심 문제와 미충족 수요"
- "핵심 고객 및 사용자"
- "기업과 고객의 접점 및 영업 방식"
- "기업과의 관계"
- "핵심 사업 수행을 위한 운영 활동"
- "핵심 자원과 경쟁력"
- "core business keyword"
- "platform business"
- "기업 본체"

[출력 규칙]
- JSON ONLY
- title은 지정된 영어 타이틀 유지. 단 core는 실제 핵심 사업 키워드, company는 실제 기업명 사용
- bullets는 1~2개
- 모든 bullet은 12자 이내를 우선

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
    normalized: Dict[str, Dict[str, Any]] = {}
    for key, default in _build_default_node_specs(company_name, bmc_data).items():
        spec = dict(data.get(key, {}) or {})
        title = clean_korean_label(spec.get("title", ""), fallback=default["title"])
        bullets = _compact_items(spec.get("bullets", []), fallback_items=default["bullets"], limit=2)
        normalized[key] = {"title": title, "bullets": bullets}
    return normalized


def _format_node_specs(specs: Dict[str, Dict[str, Any]]) -> str:
    order = [
        ("problem", "상단-좌측 Problem"),
        ("target", "상단-중앙 Target"),
        ("channel", "상단-우측 Channel"),
        ("partner", "중단-좌측 Partner"),
        ("core", "중단-중앙 Core"),
        ("operating", "중단-우측 Operating"),
        ("value", "하단-좌측 Value Proposition"),
        ("company", "하단-중앙 Company"),
        ("moat", "하단-우측 Moat"),
    ]
    lines = []
    for key, label in order:
        spec = specs.get(key, {})
        title = clean_korean_label(spec.get("title", ""))
        bullets = [clean_korean_label(x) for x in spec.get("bullets", []) if clean_korean_label(x)]
        bullet_text = ", ".join(bullets[:2]) if bullets else "-"
        lines.append(f"- [{label}] title: {title} / bullets: {bullet_text}")
    return "\n".join(lines)


def _compact_items(values: List[Any], fallback_items: List[str], limit: int = 2) -> List[str]:
    items: List[str] = []
    for value in values or []:
        cleaned = _short_phrase(value)
        if cleaned and cleaned not in items:
            items.append(cleaned)
        if len(items) >= limit:
            break
    return items or fallback_items[:limit]


def _problem_items(bmc_data: Dict[str, Any], archetype: str) -> List[str]:
    summary = dict(bmc_data.get("strategic_summary", {}) or {})
    sources = [summary.get("problem", ""), summary.get("status_quo", "")]
    relationship_items = (bmc_data.get("business_model_canvas", {}) or {}).get("customer_relationships", [])
    relationship_items = [
        item for item in relationship_items
        if not any(token in str(item) for token in ["파트너십", "유지보수", "업그레이드", "데이터 기반", "맞춤형"])
    ]
    sources.extend(relationship_items)
    items: List[str] = []
    for value in sources:
        cleaned = _problem_phrase(value, archetype)
        if not cleaned or _looks_like_solution_phrase(cleaned):
            continue
        if cleaned not in items:
            items.append(cleaned)
        if len(items) >= 2:
            break
    return items or ["시장 비효율", "기존 대안 한계"]


def _company_items(bmc_data: Dict[str, Any], company_name: str) -> List[str]:
    revenue_streams = [_financial_phrase(x, revenue=True) for x in (bmc_data.get("business_model_canvas", {}) or {}).get("revenue_streams", [])]
    revenue_streams = [x for x in revenue_streams if x][:1] or ["수익 수취"]
    cost_streams = [_financial_phrase(x, revenue=False) for x in (bmc_data.get("business_model_canvas", {}) or {}).get("cost_structure", [])]
    cost_streams = [x for x in cost_streams if x][:1] or ["비용 집행"]
    items = revenue_streams + cost_streams
    if len(items) < 2 and company_name:
        items.append("사업 운영")
    return items[:2]


def _short_phrase(value: Any, max_len: int = 12) -> str:
    text = clean_korean_label(value)
    if not text:
        return ""
    text = (
        text.replace("시장의 ", "")
        .replace("핵심 ", "")
        .replace("고객 ", "")
        .replace("사용자 ", "")
        .replace("및 ", "")
        .replace("/", "·")
    ).strip(" ,.-")
    if len(text) <= max_len:
        return text
    for sep in [",", "·", "(", ")", "/", "및"]:
        if sep in text:
            candidate = clean_korean_label(text.split(sep)[0])
            if candidate and len(candidate) <= max_len:
                return candidate
    return text[:max_len].rstrip()


def _problem_phrase(value: Any, archetype: str) -> str:
    text = clean_korean_label(value)
    if not text:
        return ""
    text = (
        text.replace("소비자는 ", "")
        .replace("고객은 ", "")
        .replace("소비자가 ", "")
        .replace("고객이 ", "")
        .replace("과정에서 ", "")
        .replace("겪습니다", "")
        .replace("느낍니다", "")
        .replace("원한다", "")
        .replace("원합니다", "")
        .replace("당연하게 여깁니다", "")
    )
    for keyword in ["인력난", "운영 비효율", "서비스 비효율", "불확실성", "비교 피로", "탐색 피로", "가격 불신", "신뢰 부족", "검색 비용", "선택 어려움"]:
        if keyword in text:
            return keyword
    if any(token in text for token in ["인력", "노동력"]) and any(token in text for token in ["부담", "확보", "관리"]):
        return "인력 운영 부담"
    if "비효율" in text and any(token in text for token in ["운영", "서비스"]):
        return "운영 비효율"
    if archetype == "brand_consumer" and "감성" in text and any(token in text for token in ["미흡", "부족", "결여"]):
        return "감성 경험 부족"
    if archetype == "brand_consumer" and "직접 바르는" in text and any(token in text for token in ["한계", "부족", "미흡"]):
        return "직접 사용 경험 부재"
    if archetype == "brand_consumer" and any(token in text for token in ["젤 네일", "네일팁", "간편함"]):
        return "간편함 편중"
    if "편의성" in text and any(token in text for token in ["주류", "중심", "편중"]):
        return "편의성 편중"
    if "검색" in text and "비교" in text:
        return "검색·비교 피로"
    if "가격" in text and any(token in text for token in ["신뢰", "의심", "불신"]):
        return "가격 신뢰 부족"
    for chunk in [x.strip() for x in text.replace("와 같은", ",").replace("로 인해", ",").split(",")]:
        candidate = _short_phrase(chunk, max_len=14)
        if candidate and candidate not in {"기존", "소비자", "고객", "커머스"} and not _looks_like_solution_phrase(candidate):
            return candidate
    return _short_phrase(text, max_len=14)


def _value_phrase(value: Any, archetype: str) -> str:
    text = clean_korean_label(value)
    if not text:
        return ""
    if archetype == "robotics_b2b" and ("인력난" in text or "인건비" in text):
        return "인력난 해소"
    if archetype == "robotics_b2b" and "품질" in text and any(token in text for token in ["일관", "균일"]):
        return "품질 일관성"
    if archetype == "brand_consumer" and ("포스트 걸코어" in text or "브랜드 미학" in text):
        return "브랜드 미학"
    if archetype == "brand_consumer" and ("감성" in text or "미적 경험" in text):
        return "감성적 사용 경험"
    if archetype == "brand_consumer" and "프리미엄" in text and any(token in text for token in ["네일", "색조", "뷰티"]):
        return "프리미엄 뷰티"
    if archetype == "robotics_b2b" and ("무인 운영" in text or "24시간" in text):
        return "무인 운영 효율"
    if archetype == "robotics_b2b" and "공간 가치" in text:
        return "공간 가치 향상"
    if "불확실성" in text:
        return "불확실성 해소"
    if "신뢰" in text:
        return "가격 신뢰 제공"
    if "앱테크" in text:
        return "앱테크 결합"
    if "저렴" in text or "합리적 가격" in text:
        return "합리적 가격"
    if "탐색" in text or "비교" in text:
        return "탐색 부담 완화"
    if "서비스" in text and len(text) <= 6:
        return ""
    return _short_phrase(text, max_len=14)


def _core_phrase(value: Any, archetype: str) -> str:
    text = clean_korean_label(value)
    if not text:
        return ""
    if "서비스 품질" in text:
        return ""
    if any(token in text for token in ["합리적 가격", "저렴", "신뢰", "불확실성", "탐색", "비교", "인력난", "인건비", "효율"]):
        return ""
    if archetype == "robotics_b2b" and "AI 로봇" in text:
        return "AI 로봇 솔루션"
    if archetype == "robotics_b2b" and "서비스 로봇" in text:
        return "서비스 로봇"
    if archetype == "brand_consumer" and "브랜드" in text:
        return "브랜드 플랫폼"
    if "플랫폼" in text:
        return "커머스 플랫폼"
    if "추천" in text:
        return "추천 엔진"
    if "직거래" in text:
        return "직거래 구조"
    if "개발 운영" in text:
        return "플랫폼 운영"
    if "판매자 유치" in text:
        return "판매자 네트워크"
    return _short_phrase(text, max_len=14)


def _financial_phrase(value: Any, revenue: bool) -> str:
    text = clean_korean_label(value)
    if not text:
        return ""
    if revenue:
        if "수수료" in text:
            return "수수료 수취"
        if "광고" in text:
            return "광고 수익"
        if "구독" in text:
            return "구독 매출"
        if "라이선스" in text:
            return "라이선스 매출"
        return "수익 수취"
    if any(token in text for token in ["개발", "운영", "인건비"]):
        return "운영비 집행"
    if any(token in text for token in ["결제", "정산", "PG"]):
        return "결제비 집행"
    if any(token in text for token in ["마케팅", "광고"]):
        return "마케팅비 집행"
    if any(token in text for token in ["인프라", "클라우드", "서버"]):
        return "인프라비 집행"
    return "비용 집행"


def _looks_like_solution_phrase(text: str) -> bool:
    return any(
        token in text
        for token in ["추천", "서비스", "플랫폼", "지원", "제공", "운영", "프로모션", "커뮤니티", "게임", "데이터"]
    )


def _join_flow_labels(flows: List[Dict[str, str]], fallback: str) -> str:
    labels = []
    for flow in flows or []:
        label = clean_korean_label(flow.get("label", ""))
        if label and label not in labels:
            labels.append(label)
        if len(labels) >= 4:
            break
    return ", ".join(labels) if labels else fallback


def _join_validated_labels_by_type(validated_flows: List[Dict[str, str]], flow_type: str, fallback: str) -> str:
    labels = []
    for flow in validated_flows or []:
        if flow.get("type") != flow_type:
            continue
        label = _short_phrase(flow.get("label", ""), max_len=14)
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
            return _balanced_role_flows(repaired)
    return _balanced_role_flows(draft_flows)


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
    return flows, ambiguous[:6]


def _infer_role_flow(flow_type: str, label: str, bmc_data: Dict[str, Any]) -> Optional[Dict[str, str]]:
    bmc = bmc_data.get("business_model_canvas", {}) or {}
    revenue_text = " ".join(bmc.get("revenue_streams", []))
    cost_text = " ".join(bmc.get("cost_structure", []))
    channel_text = " ".join(bmc.get("channels", []))
    relation_text = " ".join(bmc.get("customer_relationships", []))
    resource_text = " ".join(bmc.get("key_resources", []))
    activity_text = " ".join(bmc.get("key_activities", []))

    if flow_type == "정보":
        if "판매 데이터" in label:
            return {"from": "커뮤니티 및 채널", "to": "코어 플랫폼", "label": label}
        if any(token in label for token in ["사용", "요청", "문의", "입력", "행동"]):
            return {"from": "타겟 고객", "to": "코어 플랫폼", "label": label}
        if any(token in label for token in ["공급", "상품", "재고", "판매", "셀러"]):
            return {"from": "핵심 파트너", "to": "코어 플랫폼", "label": label}
        if any(token in label for token in ["도입", "리드", "채널", "영업"]):
            return {"from": "커뮤니티 및 채널", "to": "코어 플랫폼", "label": label}
        if any(token in label for token in ["모델", "원천", "기술", "데이터", "API", "연동"]) or any(
            token in label for token in ["모델", "원천", "기술", "데이터"]
        ):
            return {"from": "핵심 자원", "to": "코어 플랫폼", "label": label}
        return None

    if flow_type == "돈":
        if any(token in label for token in ["유통 수수료", "입점 수수료", "채널 수수료"]):
            return {"from": "기업 본체", "to": "커뮤니티 및 채널", "label": label}
        if any(token in label for token in ["판매 대금", "구축비", "유지보수", "이용료", "구독", "SaaS"]):
            return {"from": "타겟 고객", "to": "기업 본체", "label": label}
        if any(token in label for token in ["결제", "PG", "정산"]):
            return {"from": "기업 본체", "to": "핵심 파트너", "label": label}
        if any(token in label for token in ["마케팅", "광고"]) and "비용" in label:
            return {"from": "기업 본체", "to": "커뮤니티 및 채널", "label": label}
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
        if any(token in label for token in ["유지보수", "설치", "구축"]):
            return {"from": "코어 플랫폼", "to": "타겟 고객", "label": label}
        if any(token in label for token in ["인프라", "클라우드", "호스팅", "서버"]):
            return {"from": "핵심 자원", "to": "코어 플랫폼", "label": label}
        if any(token in label for token in ["공급", "입점", "상품", "판매자", "셀러"]):
            return {"from": "핵심 파트너", "to": "코어 플랫폼", "label": label}
        if any(token in label for token in ["솔루션", "도입", "채널", "영업", "제휴"]):
            return {"from": "코어 플랫폼", "to": "커뮤니티 및 채널", "label": label}
        if any(token in label for token in ["API", "연동"]) and "도입" not in label:
            return {"from": "핵심 파트너", "to": "코어 플랫폼", "label": label}
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


def _balanced_role_flows(flows: List[Dict[str, str]], limit: int = 8) -> List[Dict[str, str]]:
    deduped = _dedupe_role_flows(flows)
    groups = {
        "정보": [flow for flow in deduped if flow.get("type") == "정보"],
        "돈": [flow for flow in deduped if flow.get("type") == "돈"],
        "서비스": [flow for flow in deduped if flow.get("type") == "서비스"],
    }
    selected: List[Dict[str, str]] = []
    # First pass: keep up to 2 from each type so one category does not starve others.
    for flow_type in ["정보", "돈", "서비스"]:
        selected.extend(groups[flow_type][:2])
    if len(selected) < limit:
        leftovers: List[Dict[str, str]] = []
        for flow_type in ["서비스", "정보", "돈"]:
            leftovers.extend(groups[flow_type][2:])
        for flow in leftovers:
            if len(selected) >= limit:
                break
            selected.append(flow)
    return selected[:limit]


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
