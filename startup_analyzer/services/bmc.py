import json
from typing import Any, Dict, List

from google import genai
from google.genai import types

from startup_analyzer.services.analysis import MODEL_NAME
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

    output["top_layer"] = [
        clean_korean_label(x) for x in normalize_text_list(output.get("top_layer", []), limit=3)
    ]
    output["left_actors"] = [
        clean_korean_label(x) for x in normalize_text_list(output.get("left_actors", []), limit=4)
    ]
    output["right_actors"] = [
        clean_korean_label(x) for x in normalize_text_list(output.get("right_actors", []), limit=4)
    ]
    output["middle_layer"] = clean_korean_label(output.get("middle_layer", ""), fallback="핵심 플랫폼")

    if not any(output["top_layer"]):
        output["top_layer"] = output["business_model_canvas"]["customer_segments"][:1] or ["핵심 고객"]
    if not any(output["left_actors"]):
        output["left_actors"] = output["business_model_canvas"]["key_partnerships"][:1] or ["핵심 파트너"]
    if not any(output["right_actors"]):
        right_candidates = output["business_model_canvas"]["channels"] or output["business_model_canvas"]["customer_relationships"]
        output["right_actors"] = right_candidates[:1] or ["핵심 채널"]

    output["money_flows"] = _normalize_flow_items(output.get("money_flows", []))
    output["information_flows"] = _normalize_flow_items(output.get("information_flows", []))
    output["service_flows"] = _normalize_flow_items(output.get("service_flows", []))

    _ensure_bottom_relationship(output, company_name)
    return output


def _ensure_bottom_relationship(data: Dict[str, Any], company_name: str):
    company_label = clean_korean_label(company_name, fallback=company_name)
    center_label = data.get("middle_layer", "핵심 플랫폼")

    all_flows = data.get("money_flows", []) + data.get("information_flows", []) + data.get("service_flows", [])
    has_bottom_flow = any(company_label and (flow.get("from") == company_label or flow.get("to") == company_label) for flow in all_flows)
    if has_bottom_flow:
        return

    data["service_flows"].append({"from": company_label, "to": center_label, "label": "운영"})
    data["information_flows"].append({"from": center_label, "to": company_label, "label": "운영 데이터"})


def build_bmc_and_diagram_data(
    client: genai.Client,
    company_name: str,
    ceo_name: str,
    facts: str,
    profile: Dict[str, Any],
    keywords: List[str],
) -> Dict[str, Any]:
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
{json.dumps(profile, ensure_ascii=False, indent=2)}

[핵심 목표]
1. 정식 Business Model Canvas 9블록을 완성한다.
2. SVG 생태계형 BM 다이어그램에 필요한 노드와 흐름을 만든다.
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
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="text/plain"),
    )
    raw_text = (response.text or "").strip()

    try:
        data = extract_json(raw_text)
    except Exception:
        repaired = repair_json_with_model(client, MODEL_NAME, raw_text, schema_hint=BMC_SCHEMA_HINT)
        data = extract_json(repaired)

    return ensure_bmc_shape(data, company_name=company_name)
