import json
from typing import Any, Dict, List

from google import genai
from google.genai import types

from startup_analyzer.services.analysis import TEXT_MODEL
from startup_analyzer.utils.json_utils import extract_json, repair_json_with_model
from startup_analyzer.utils.text import clean_korean_label, normalize_text_list


BMC_SCHEMA_HINT = """
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

EXCLUDED_PARTNER_TERMS = ["투자", "벤처캐피탈", "VC", "엑셀러레이터", "인베스트", "펀드"]
GENERIC_CHANNEL_TERMS = ["온라인", "플랫폼", "웹사이트", "직접 영업", "영업", "커뮤니티"]


def _normalize_flow_items(items: Any) -> List[Dict[str, str]]:
    normalized = []
    for item in items or []:
        if isinstance(item, dict):
            normalized.append(
                {
                    "from": clean_korean_label(item.get("from", "")),
                    "to": clean_korean_label(item.get("to", "")),
                    "label": clean_korean_label(item.get("label", "")),
                }
            )
        else:
            text = clean_korean_label(item)
            if text:
                normalized.append({"from": "", "to": "", "label": text})
        if len(normalized) >= 8:
            break
    return [item for item in normalized if item.get("label")]


def ensure_bmc_shape(data: Dict[str, Any], company_name: str = "") -> Dict[str, Any]:
    output = dict(data or {})
    output["bm_type"] = clean_korean_label(output.get("bm_type", ""), fallback="플랫폼형")
    output["bmc_summary"] = str(output.get("bmc_summary", "")).strip()

    summary = dict(output.get("strategic_summary", {}) or {})
    output["strategic_summary"] = {
        "problem": str(summary.get("problem", "")).strip(),
        "status_quo": str(summary.get("status_quo", "")).strip(),
        "our_solution": str(summary.get("our_solution", "")).strip(),
    }

    bmc = dict(output.get("business_model_canvas", {}) or {})
    output["business_model_canvas"] = {
        "customer_segments": [clean_korean_label(x) for x in normalize_text_list(bmc.get("customer_segments", []))],
        "value_propositions": [clean_korean_label(x) for x in normalize_text_list(bmc.get("value_propositions", []))],
        "channels": [clean_korean_label(x) for x in normalize_text_list(bmc.get("channels", []))],
        "customer_relationships": [clean_korean_label(x) for x in normalize_text_list(bmc.get("customer_relationships", []))],
        "revenue_streams": [clean_korean_label(x) for x in normalize_text_list(bmc.get("revenue_streams", []))],
        "key_resources": [clean_korean_label(x) for x in normalize_text_list(bmc.get("key_resources", []))],
        "key_activities": [clean_korean_label(x) for x in normalize_text_list(bmc.get("key_activities", []))],
        "key_partnerships": [clean_korean_label(x) for x in normalize_text_list(bmc.get("key_partnerships", []))],
        "cost_structure": [clean_korean_label(x) for x in normalize_text_list(bmc.get("cost_structure", []))],
    }

    output["top_layer"] = [clean_korean_label(x) for x in normalize_text_list(output.get("top_layer", []), limit=3)]
    output["left_actors"] = [clean_korean_label(x) for x in normalize_text_list(output.get("left_actors", []), limit=4)]
    output["right_actors"] = [clean_korean_label(x) for x in normalize_text_list(output.get("right_actors", []), limit=4)]
    output["middle_layer"] = clean_korean_label(output.get("middle_layer", ""), fallback="핵심 플랫폼")

    output["top_layer"] = _derive_top_layer(output)
    output["left_actors"] = _derive_left_actors(output)
    output["right_actors"] = _derive_right_actors(output)

    output["money_flows"] = _normalize_flow_items(output.get("money_flows", []))
    output["information_flows"] = _normalize_flow_items(output.get("information_flows", []))
    output["service_flows"] = _normalize_flow_items(output.get("service_flows", []))

    _refine_diagram_structure(output, company_name)
    return output


def _refine_diagram_structure(data: Dict[str, Any], company_name: str):
    company_label = clean_korean_label(company_name, fallback=company_name)
    center_label = data.get("middle_layer", "핵심 플랫폼")
    top_label = data.get("top_layer", ["핵심 고객"])[0]
    left_label = data.get("left_actors", ["핵심 파트너"])[0]
    right_label = data.get("right_actors", ["핵심 채널"])[0]

    revenue_label = _choose_revenue_label(data.get("business_model_canvas", {}).get("revenue_streams", []))
    cost_label = _choose_cost_label(data.get("business_model_canvas", {}).get("cost_structure", []), left_label)
    info_top_label = _choose_info_top_label(data)
    info_left_label = _choose_info_left_label(data)
    info_right_label = _choose_info_right_label(right_label)
    service_top_label = _choose_service_top_label(data)
    service_right_label = _choose_service_right_label(right_label)

    data["money_flows"] = [
        {"from": top_label, "to": company_label, "label": revenue_label},
        {"from": company_label, "to": left_label, "label": cost_label},
    ]

    info_flows = [
        {"from": top_label, "to": center_label, "label": info_top_label},
        {"from": left_label, "to": center_label, "label": info_left_label},
        {"from": center_label, "to": company_label, "label": "운영 데이터"},
    ]
    if right_label:
        info_flows.append({"from": right_label, "to": center_label, "label": info_right_label})
    data["information_flows"] = info_flows

    service_flows = [
        {"from": center_label, "to": top_label, "label": service_top_label},
        {"from": company_label, "to": center_label, "label": "플랫폼 운영"},
        {"from": left_label, "to": center_label, "label": "기술 연동"},
    ]
    if right_label:
        service_flows.append({"from": center_label, "to": right_label, "label": service_right_label})
    data["service_flows"] = service_flows


def _choose_revenue_label(revenue_streams: List[str]) -> str:
    joined = " ".join(revenue_streams)
    if "구독" in joined:
        return "구독료"
    if "수수료" in joined:
        return "수수료"
    if "라이선스" in joined:
        return "라이선스비"
    if "컨설팅" in joined:
        return "컨설팅료"
    return "이용료"


def _choose_cost_label(cost_structure: List[str], left_label: str) -> str:
    joined = " ".join(cost_structure)
    if "모델" in left_label or "AI" in left_label:
        return "모델 비용"
    if "데이터" in left_label:
        return "데이터 비용"
    if "인프라" in joined:
        return "인프라 비용"
    if "라이선스" in joined:
        return "라이선스비"
    if "데이터" in joined:
        return "데이터 비용"
    return "운영 비용"


def _choose_info_top_label(data: Dict[str, Any]) -> str:
    joined = " ".join(data.get("business_model_canvas", {}).get("customer_relationships", []))
    if "데이터" in joined:
        return "사용 데이터"
    return "요청 정보"


def _choose_info_left_label(data: Dict[str, Any]) -> str:
    joined = " ".join(data.get("business_model_canvas", {}).get("key_resources", []))
    if "모델" in joined or "AI" in joined:
        return "모델 정보"
    if "데이터" in joined:
        return "데이터 정보"
    return "기술 정보"


def _choose_info_right_label(right_label: str) -> str:
    if any(token in right_label for token in ["도입", "구축", "SI", "공급", "파트너"]):
        return "도입 정보"
    return "채널 정보"


def _choose_service_top_label(data: Dict[str, Any]) -> str:
    joined = " ".join(data.get("business_model_canvas", {}).get("value_propositions", []))
    if "보안" in joined:
        return "보안 서비스"
    if "분석" in joined:
        return "분석 서비스"
    if "자동화" in joined:
        return "자동화 기능"
    return "핵심 서비스"


def _choose_service_right_label(right_label: str) -> str:
    if any(token in right_label for token in ["도입", "구축", "SI", "공급", "파트너"]):
        return "솔루션 제공"
    return "채널 지원"


def build_bmc_and_diagram_data(
    client: genai.Client,
    company_name: str,
    ceo_name: str,
    facts: str,
    profile: Dict[str, Any],
    keywords: List[str],
) -> Dict[str, Any]:
    filtered_profile = {
        "problem_definition": profile.get("problem_definition", ""),
        "solution_value_prop": profile.get("solution_value_prop", ""),
        "revenue_model_type": profile.get("revenue_model_type", ""),
        "product_core_features": profile.get("product_core_features", []),
        "core_tech_moat": profile.get("core_tech_moat", ""),
        "ceo_vision_summary": profile.get("ceo_vision_summary", ""),
        "industry_keywords": profile.get("industry_keywords", []),
    }

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
{json.dumps(filtered_profile, ensure_ascii=False, indent=2)}

[핵심 목표]
1. 정식 Business Model Canvas 9블록을 완성한다.
2. BM 다이어그램 PNG 생성에 필요한 노드와 흐름을 만든다.
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
- 한자, 일본어, 번역투 표현 금지
- 회사명과 중앙 플랫폼/제품의 관계를 흐름으로 분명히 드러낼 것
- 동일한 두 노드 사이에 여러 흐름이 생길 수 있으므로, 흐름 라벨은 서로 다른 의미로 작성
- bmc_summary는 2~3문장 분량의 한국어 문단
- strategic_summary의 각 필드는 한 문장으로 작성

[다이어그램 설계 원칙]
- 상단 노드는 핵심 고객 또는 수요 주체
- 좌측 노드는 공급/제휴/인프라 측 핵심 주체
- 중앙 노드는 플랫폼/솔루션/핵심 운영 엔진
- 우측 노드는 유통/채널/도입 조직/수요처 중 핵심 주체
- 하단 노드는 회사명 자체로 해석될 수 있도록 작성
- 흐름은 돈, 정보, 서비스/자산의 의미가 드러나야 함

[출력 스키마]
{BMC_SCHEMA_HINT}
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
        repaired = repair_json_with_model(client, TEXT_MODEL, raw_text, schema_hint=BMC_SCHEMA_HINT)
        data = extract_json(repaired)

    return ensure_bmc_shape(data, company_name=company_name)


def _derive_top_layer(data: Dict[str, Any]) -> List[str]:
    customer_segments = data.get("business_model_canvas", {}).get("customer_segments", [])
    for item in customer_segments + data.get("top_layer", []):
        label = clean_korean_label(item)
        if label:
            return [label]
    return ["핵심 고객"]


def _derive_left_actors(data: Dict[str, Any]) -> List[str]:
    candidates = (
        data.get("business_model_canvas", {}).get("key_partnerships", [])
        + data.get("left_actors", [])
        + data.get("business_model_canvas", {}).get("key_resources", [])
    )
    for item in candidates:
        label = clean_korean_label(item)
        if not label:
            continue
        if any(term.lower() in label.lower() for term in EXCLUDED_PARTNER_TERMS):
            continue
        return [label]
    return ["핵심 파트너"]


def _derive_right_actors(data: Dict[str, Any]) -> List[str]:
    candidates = data.get("business_model_canvas", {}).get("channels", []) + data.get("right_actors", [])
    for item in candidates:
        label = clean_korean_label(item)
        if not label:
            continue
        if any(term in label for term in GENERIC_CHANNEL_TERMS):
            continue
        return [label]
    if candidates:
        label = clean_korean_label(candidates[0], fallback="핵심 채널")
        if label and not label.endswith(("채널", "파트너", "공급사", "고객사", "리셀러")):
            label = f"{label} 채널"
        return [label]
    return ["핵심 채널"]
