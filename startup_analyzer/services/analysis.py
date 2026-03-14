import os
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from startup_analyzer.utils.json_utils import extract_json, repair_json_with_model


TEXT_MODEL = "gemini-2.5-flash"


PROFILE_SCHEMA_HINT = """
{
  "problem_definition": "",
  "solution_value_prop": "",
  "revenue_model_type": "",
  "product_core_features": [],
  "core_tech_moat": "",
  "ceo_vision_summary": "",
  "org_culture_biz_focus": "",
  "recent_news_summary": "",
  "industry_keywords": []
}
""".strip()


def get_gemini_api_key() -> Optional[str]:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def build_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def build_google_tool() -> types.Tool:
    return types.Tool(google_search=types.GoogleSearch())


def gather_company_facts(
    client: genai.Client,
    company_name: str,
    ceo_name: str,
    raw_text: str = "",
) -> str:
    google_tool = build_google_tool()
    prompt = f"""
회사명 {company_name}, 대표자 {ceo_name}에 대한 사실 기반 정보를 Google 검색으로 수집하라.

[규칙]
- 검증된 사실만 작성
- 대표자 인터뷰/발언이 있으면 반드시 포함
- 추측, 요약, 해석 금지
- JSON 금지
- 텍스트만 출력
"""

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[google_tool],
            response_mime_type="text/plain",
        ),
    )
    facts = (response.text or "").strip()
    if raw_text.strip():
        facts = f"{facts}\n\n[사용자 보조 텍스트]\n{raw_text.strip()}"
    return facts


def generate_company_profile(
    client: genai.Client,
    company_name: str,
    facts: str,
) -> Dict[str, Any]:
    prompt = f"""
아래는 {company_name}에 관한 사실 기반 정보이다:
{facts}

아래 기준에 따라 기업 분석 JSON만 생성하라.

[기업 분석 지침]
- 객관적, 분석적 전문가 문체
- 특수문자("*","**","~") 금지
- 각 항목 최소 120자 이상
- 기업명 기반 뻔한 설명 금지
- 대표자 비전은 공신력 있는 출처 기반
- 조직문화는 채용사이트 표현 금지
- 불확실한 정보는 확인 불가
- 추정은 (추정됨) 또는 (예상됨) 명시
- 광고성 표현 금지
- 문자열 값 내부 큰따옴표 사용 금지
- JSON ONLY

[출력 스키마]
{PROFILE_SCHEMA_HINT}
"""

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="text/plain"),
    )
    raw_text = (response.text or "").strip()

    try:
        return extract_json(raw_text)
    except Exception:
        fixed_text = repair_json_with_model(client, TEXT_MODEL, raw_text, schema_hint=PROFILE_SCHEMA_HINT)
        return extract_json(fixed_text)
