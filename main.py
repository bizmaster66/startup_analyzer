from google import genai
from google.genai import types
from utils import extract_json_from_text, extract_industry_keywords


# -------------------------------------------------------------
# 기업 프로필 생성
# -------------------------------------------------------------
def generate_company_profile(api_key, model_name, company_name, ceo_name, raw_text):
    client = genai.Client(api_key=api_key)
    google_tool = types.Tool(google_search=types.GoogleSearch())

    prompt = f"""
    당신은 스타트업 분석 전문가입니다.
    회사명: {company_name}
    대표자: {ceo_name}

    규칙:
    - Google Search 기반 정보만 사용
    - 추론 시 '(추정됨)' '(예상됨)' 표기
    - 근거 없으면 '확인 불가'
    - JSON Only

    JSON 형식:
    {{
        "problem_definition": "",
        "solution_value_prop": "",
        "revenue_model_type": "",
        "product_core_features": [],
        "core_tech_moat": "",
        "ceo_vision_summary": "",
        "org_culture_biz_focus": "",
        "recent_news_summary": "",
        "industry_keywords": []
    }}
    """

    try:
        cfg = types.GenerateContentConfig(
            tools=[google_tool],
            response_mime_type="application/json"
        )

        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=cfg
        )
        text = resp.text

    except:
        cfg = types.GenerateContentConfig(response_mime_type="application/json")
        fallback = raw_text + "\n" + prompt
        resp = client.models.generate_content(
            model=model_name,
            contents=fallback,
            config=cfg
        )
        text = resp.text

    return extract_json_from_text(text)


# -------------------------------------------------------------
# 산업 리포트 생성
# -------------------------------------------------------------
def generate_industry_report(api_key, model_name, keywords):
    client = genai.Client(api_key=api_key)
    google_tool = types.Tool(google_search=types.GoogleSearch())

    kw_str = ", ".join(keywords)

    prompt = f"""
    산업 키워드: {kw_str}

    다음 기준으로 산업 리포트 작성:
    - 대한민국 vs 글로벌 시장 규모 비교
    - 성장률 / 주요 기업 / 경쟁 구도
    - 정책 / 규제 / 리스크 요인
    - 투자 동향
    - 3~5년 전망 (추정됨/예상됨)
    - 수치 없으면 '관련 데이터 없음'
    - 전문적인 문체

    출력은 일반 텍스트로 작성할 것.
    """

    cfg = types.GenerateContentConfig(
        tools=[google_tool],
        response_mime_type="text/plain"    # ★ 수정됨 (text/markdown 금지)
    )

    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=cfg
    )

    return resp.text
