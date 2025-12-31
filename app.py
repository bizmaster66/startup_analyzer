import os
import re
import json
import base64
import datetime
from typing import Optional, List, Dict, Any

import streamlit as st
from google import genai
from google.genai import types


# ======================= API KEY =======================
def get_gemini_api_key() -> Optional[str]:
    """
    우선순위:
    1) Streamlit Secrets: GEMINI_API_KEY, GOOGLE_API_KEY
    2) Environment Variables: GEMINI_API_KEY, GOOGLE_API_KEY
    키가 없으면 안내 메시지 출력 후 None 반환
    """
    # 1) Streamlit Secrets (배포용)
    try:
        key = st.secrets.get("GEMINI_API_KEY", None) or st.secrets.get("GOOGLE_API_KEY", None)
        if key:
            return str(key).strip()
    except Exception:
        # 로컬에서 secrets.toml이 없으면 StreamlitSecretNotFoundError가 발생할 수 있음
        pass

    # 2) Environment Variables (로컬용)
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return str(key).strip()

    # 3) 안내
    st.error(
        "Gemini API Key가 설정되지 않았습니다.\n\n"
        "로컬 실행:\n"
        "  - 터미널에서 `export GEMINI_API_KEY=\"YOUR_KEY\"` 설정 후 실행\n\n"
        "Streamlit Cloud 배포:\n"
        "  - App settings → Secrets 에 아래처럼 추가\n"
        "    GEMINI_API_KEY=\"YOUR_KEY\"\n"
        "    GOOGLE_API_KEY=\"YOUR_KEY\" (선택)"
    )
    return None


# ======================= JSON Extract & Repair =======================
def _escape_inner_quotes_heuristic(s: str) -> str:
    """
    휴리스틱으로 문자열 내부의 비정상 " 를 \\" 로 이스케이프.
    - JSON 문자열 시작/종료를 추적
    - 문자열 내부에서 등장하는 " 중 '닫힘'이 아니라면 \\" 처리
    """
    out = []
    in_str = False
    esc = False
    i = 0
    n = len(s)

    def next_non_space(idx: int) -> str:
        j = idx
        while j < n and s[j].isspace():
            j += 1
        return s[j] if j < n else ""

    while i < n:
        ch = s[i]

        if esc:
            out.append(ch)
            esc = False
            i += 1
            continue

        if ch == "\\":
            out.append(ch)
            esc = True
            i += 1
            continue

        if ch == '"':
            if not in_str:
                # 문자열 시작
                in_str = True
                out.append(ch)
            else:
                # 문자열 내부에서 만난 "
                # 닫힘 따옴표면 다음 non-space가 , } ] 등으로 이어지는 경우가 많음
                nxt = next_non_space(i + 1)
                if nxt in [",", "}", "]"]:
                    # 문자열 닫힘
                    in_str = False
                    out.append(ch)
                else:
                    # 문자열 내부 따옴표로 보고 이스케이프
                    out.append('\\"')
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def extract_json(text: str) -> Dict[str, Any]:
    """
    1) 코드펜스 제거
    2) 가장 바깥 JSON 블록({ ... })만 추출
    3) 1차 json.loads 시도
    4) 실패 시 휴리스틱 이스케이프 후 재시도
    """
    if not text:
        raise ValueError("Empty response")

    cleaned = text.replace("```json", "").replace("```", "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON block not found")

    raw = cleaned[start:end + 1]

    # 1차 파싱
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2차 휴리스틱 복구
    repaired = _escape_inner_quotes_heuristic(raw)
    return json.loads(repaired)


def repair_json_with_model(client: genai.Client, raw_text: str) -> str:
    """
    모델에게 '표준 JSON'으로만 정정하도록 재요청.
    """
    fix_prompt = (
        "아래 출력은 JSON 형식이 깨져 있습니다. 내용을 최대한 동일하게 유지하되,\n"
        "반드시 '표준 JSON'으로만 수정해서 JSON만 출력하세요.\n\n"
        "[규칙]\n"
        "- 문자열 내부에 큰따옴표(\")가 필요하면 반드시 \\\" 로 이스케이프하거나, 인용부호 없이 서술하세요.\n"
        "- 코드펜스(```), 설명 문장, 주석 금지. JSON ONLY.\n"
        "- 키 이름/구조는 유지하고, 값만 JSON 문법에 맞게 고치세요.\n\n"
        "[원본]\n"
        f"{raw_text}\n"
    )

    fix_resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=fix_prompt,
        config=types.GenerateContentConfig(response_mime_type="text/plain"),
    )
    return (fix_resp.text or "").strip()


# ======================= STEP UI =======================
def render_step(step: int):
    if step == 1:
        html = """
<div style="display:flex;gap:12px;margin-bottom:24px;">
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 1 · 정보 수집</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#F1F5F9;color:#475569;">
STEP 2 · JSON 분석</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#F1F5F9;color:#475569;">
STEP 3 · 기업 분석 결과</div>
</div>
"""
    elif step == 2:
        html = """
<div style="display:flex;gap:12px;margin-bottom:24px;">
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 1 · 정보 수집</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 2 · JSON 분석</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#F1F5F9;color:#475569;">
STEP 3 · 기업 분석 결과</div>
</div>
"""
    else:
        html = """
<div style="display:flex;gap:12px;margin-bottom:24px;">
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 1 · 정보 수집</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 2 · JSON 분석</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 3 · 기업 분석 결과</div>
</div>
"""
    st.write(html, unsafe_allow_html=True)


# ======================= TILE UI =======================
def tile(title: str, body: str):
    safe = (body or "").replace("\n", "<br>")
    st.write(
        f"""
<div style="background:#E2E8F0;padding:10px 14px;border-radius:6px 6px 0 0;
border:1px solid #CBD5E1;font-weight:600;">{title}</div>""",
        unsafe_allow_html=True
    )
    st.write(
        f"""
<div style="background:white;padding:16px;border-radius:0 0 6px 6px;
border:1px solid #CBD5E1;border-top:none;font-size:14px;line-height:1.6;">
{safe}</div>""",
        unsafe_allow_html=True
    )
    st.write("<div style='height:14px;'></div>", unsafe_allow_html=True)


# ======================= DOWNLOAD =======================
def download_button(label: str, text: str, filename: str):
    b64 = base64.b64encode((text or "").encode()).decode()
    st.markdown(
        f"""<a href="data:text/markdown;base64,{b64}"
               download="{filename}"
               style="font-size:16px;color:#00D2A8;">{label}</a>""",
        unsafe_allow_html=True,
    )


# ======================= Keyword Extract =======================
def extract_keywords(profile: dict) -> List[str]:
    kws = [k for k in profile.get("industry_keywords", []) if "확인 불가" not in str(k)]
    if kws:
        return kws

    features = profile.get("product_core_features", [])
    if isinstance(features, list):
        tokens = " ".join([str(x) for x in features]).lower().split()
    else:
        tokens = str(features).lower().split()

    auto = [t for t in tokens if len(t) > 3]
    return list(set(auto))[:5] if auto else ["technology"]


# ======================= PAGE HEADER =======================
st.set_page_config(layout="wide", page_title="혁신의숲 Startup Analyzer & Report")

st.write(
    """
<div style="width:100%;padding:26px 0;text-align:center;">
    <div style="margin-bottom:4px;">
        <span style="color:#00D2A8;font-size:32px;font-weight:700;">
            혁신의숲 Startup Analyzer & Report
        </span>
    </div>
    <div>
        <span style="color:#64748B;font-size:14px;font-weight:500;">
            Powered by Mark & Company
        </span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ======================= SIDEBAR =======================
with st.sidebar:
    st.markdown("### 설정")
    st.caption("API Key는 화면 입력이 아니라 Secrets/환경변수에서 읽습니다.")
    with st.expander("로컬/배포 설정 방법", expanded=False):
        st.code(
            '로컬:\n'
            '  export GEMINI_API_KEY="YOUR_KEY"\n'
            '  python -m streamlit run app.py\n\n'
            'Streamlit Cloud Secrets:\n'
            '  GEMINI_API_KEY="YOUR_KEY"\n'
            '  GOOGLE_API_KEY="YOUR_KEY"  # 선택\n',
            language="bash",
        )
    st.markdown("---")


# ======================= INPUT FORM =======================
st.markdown("## 기업 정보 입력")

col1, col2 = st.columns(2)
with col1:
    company_name = st.text_input("기업명", placeholder="예: 마크앤컴퍼니")
with col2:
    ceo_name = st.text_input("대표자명", placeholder="예: 홍경표")

raw_text = st.text_area(
    "보조 텍스트 (뉴스/메모 등)",
    height=130,
    placeholder="기업과 관련된 기사나 참고 텍스트를 입력하세요. (선택)",
)

run = st.button("분석 실행", type="primary")


# ======================= RUN =======================
if run:
    if not company_name.strip():
        st.error("기업명을 입력해주세요.")
        st.stop()
    if not ceo_name.strip():
        st.error("대표자명을 입력해주세요.")
        st.stop()

    api_key = get_gemini_api_key()
    if not api_key:
        st.stop()

    client = genai.Client(api_key=api_key)
    google_tool = types.Tool(google_search=types.GoogleSearch())

    # ============================================================
    # STEP 1 : FACT 수집
    # ============================================================
    render_step(1)

    gather_prompt = f"""
회사명 {company_name}, 대표자 {ceo_name}에 대한 사실 기반 정보를 Google 검색으로 수집하라.

[규칙]
- 검증된 사실만 작성
- 대표자 인터뷰/발언이 있으면 반드시 포함
- 추측, 요약, 해석 금지
- JSON 금지
- 텍스트만 출력
"""

    gather_resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=gather_prompt,
        config=types.GenerateContentConfig(
            tools=[google_tool],
            response_mime_type="text/plain",
        ),
    )
    facts = (gather_resp.text or "").strip()

    # ============================================================
    # STEP 2 : 기업 분석 JSON 생성
    # ============================================================
    render_step(2)

    json_prompt = f"""
아래는 {company_name}에 관한 사실 기반 정보이다:
{facts}

아래 기준에 따라 기업 분석 JSON만 생성하라.

[기업 분석 지침]
- 객관적, 분석적 전문가 문체
- 특수문자("*","**","~") 금지
- 각 항목 최소 120자 이상
- 기업명 기반 뻔한 설명 금지
- 대표자 비전: 공신력 있는 출처 기반
- 조직문화: 채용사이트 언급 금지
- 불확실한 정보는 "확인 불가"
- 추정은 "(추정됨)" 또는 "(예상됨)" 명시
- 광고성/감성적 표현 금지
- 문자열 값 내부에 큰따옴표(") 사용 금지 (인용은 따옴표 없이 서술)
- JSON ONLY 출력

출력 형식:
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

    json_resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=json_prompt,
        config=types.GenerateContentConfig(response_mime_type="text/plain"),
    )

    raw_json_text = (json_resp.text or "").strip()

    # 1) 파싱 1차 시도(휴리스틱 복구 포함)
    try:
        profile = extract_json(raw_json_text)
    except Exception:
        # 2) 모델에게 '표준 JSON'으로 정정 재요청
        st.warning("JSON 파싱 오류가 발생해 자동 정정(Repair)을 시도합니다.")
        fixed_text = repair_json_with_model(client, raw_json_text)

        try:
            profile = extract_json(fixed_text)
        except Exception:
            st.error("JSON 자동 정정에도 실패했습니다. 아래 원본/정정본을 확인하세요.")
            st.markdown("### 원본 출력")
            st.code(raw_json_text)
            st.markdown("### 정정 시도 출력")
            st.code(fixed_text)
            st.stop()

    # ============================================================
    # STEP 3 : 기업 분석 결과 출력
    # ============================================================
    render_step(3)
    st.markdown("## 기업 분석 결과")

    tile("문제 정의", profile.get("problem_definition", ""))
    tile("솔루션 및 제공 가치", profile.get("solution_value_prop", ""))
    tile("비즈니스 모델", profile.get("revenue_model_type", ""))
    tile("핵심 기능", "<br>".join(profile.get("product_core_features", []) or []))
    tile("핵심 기술 · 경쟁력", profile.get("core_tech_moat", ""))
    tile("대표자 비전", profile.get("ceo_vision_summary", ""))
    tile("조직 · 운영 방식", profile.get("org_culture_biz_focus", ""))
    tile("최근 뉴스 요약", profile.get("recent_news_summary", ""))

    keywords = extract_keywords(profile)
    tile("산업 키워드", ", ".join(keywords))

    # ============================================================
    # 산업 리포트 생성 (요약 + 상세)
    # ============================================================
    st.markdown("## 산업 리포트 요약")

    industry_prompt_summary = f"""
산업 키워드: {", ".join(keywords)}

아래 기준에 따라 산업 '요약본'을 작성하라.

[산업 요약 지침]
- 전체 산업 분석 내용을 간결하게 요약한 버전
- 글로벌/한국 구분 없이 통합 요약
- 주요 시장동향, 투자 흐름, 주요 기업, 기술 변화, 리스크 요인 포함
- 출처 URL 반드시 포함
- URL을 검증할 수 없는 데이터는 작성 금지
- 특수문자("*","**","~") 금지
- 텍스트 ONLY
- 할루시네이션 절대 금지
"""

    summary_resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=industry_prompt_summary,
        config=types.GenerateContentConfig(response_mime_type="text/plain"),
    )
    industry_summary = (summary_resp.text or "").strip()
    tile("산업 리포트 요약", industry_summary)

    # ======================
    # 산업 상세 리포트 (다운로드 전용, 프론트 미노출)
    # ======================
    industry_prompt_detail = f"""
산업 키워드: {", ".join(keywords)}

아래 기준에 따라 산업 '상세 리포트'를 작성하라.

[산업 상세 리포트 지침]
- 글로벌 상세 시장 분석
  - 시장 규모
  - CAGR / 성장 요인
  - 경쟁 구도
  - 공급망 구조
  - 규제 영향
  - 기술 변화
  - 주요 기업
  - 향후 전망
- 한국 상세 시장 분석
  - 시장 구조
  - 정부 정책 및 규제 영향
  - 주요 기업 및 생태계
  - 투자 동향
  - 향후 전망
- 글로벌과 한국 비교 금지 (절대 금지)
- 출처 URL 반드시 포함
- URL 확인 불가한 데이터는 절대 작성 금지
- 특수문자("*","**","~") 금지
- 전문가 보고서 톤 (컨설팅 보고서)
- 텍스트 ONLY
- 할루시네이션 절대 금지
"""

    detail_resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=industry_prompt_detail,
        config=types.GenerateContentConfig(response_mime_type="text/plain"),
    )
    industry_detail = (detail_resp.text or "").strip()

    # ============================================================
    # 전체 리포트 생성 (다운로드 전용)
    # ============================================================
    st.markdown("## 전체 리포트 다운로드")

    full_prompt = f"""
아래는 {company_name}의 기업 분석 결과이다:
{json.dumps(profile, ensure_ascii=False, indent=2)}

아래는 산업 상세 리포트이다:
{industry_detail}

위 두 내용을 기반으로 전문가 분석 문체의
하나의 완전한 종합 리포트를 작성하라.

[전체 리포트 작성 규칙]
- 문단형 분석만 사용
- SWOT / 3C / 5 Forces / BCG 등 전략 프레임워크 절대 금지
- 분석은 설명식 문단만 사용
- 특수문자("*","**","~") 금지
- 기업명 기반 뻔한 설명 금지
- 대표자 비전은 공신력 있는 출처 기반으로만 작성
- 출처 + URL 반드시 포함
- URL 확인 불가한 데이터는 작성 금지
- 시장 전망·추론은 "(추정됨)" 또는 "(예상됨)"으로 명시
- JSON 언급 금지
- 텍스트 ONLY
"""

    full_resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=full_prompt,
        config=types.GenerateContentConfig(response_mime_type="text/plain"),
    )
    final_report = (full_resp.text or "").strip()

    filename = f"Full_Report_{company_name}_{datetime.date.today()}.md"
    download_button("📥 전체 리포트 다운로드", final_report, filename)

    # ============================================================
    # 산업군 A4 1~2장 상세 리포트 생성 (다운로드 전용)
    # ============================================================
    st.markdown("## 산업군 상세 리포트 다운로드")

    industry_detailed_prompt = f"""
대상 기업: {company_name}
산업 키워드: {", ".join(keywords)}

아래 목차에 따라 '해당 기업이 속한 산업군'에 대한 A4 1~2장 분량의
상세 산업 리포트를 작성하라.

[절대 규칙]
- 문단형 텍스트 ONLY
- 특수문자("*","**","~") 금지
- 할루시네이션 금지
- 데이터는 검색으로 출처 URL 검증된 내용만 사용
- URL 확인 불가한 정보는 작성 금지
- SWOT / 3C / 5 Forces 금지
- 전문가 리포트 문체
- 기업명 기반 뻔한 설명 금지

[리포트 목차]
I. 산업 개요 및 시장 현황 (Industry & Market Status)
1. 산업군 정의 및 분석 범위
2. 시장 규모 및 성장성 (출처 + URL 필수)
3. 산업의 주요 변화 동인 (Drivers)

II. 고객의 문제 및 핵심 트렌드 (Pain Points & Trends)
1. 시장의 문제점 (Pain Point)
2. 핵심 기술 및 서비스 트렌드

III. 경쟁 구도 및 스타트업의 기회 (Competition & Opportunity)
1. 핵심 경쟁사 분석 (출처 + URL 필수)
2. 스타트업의 차별화 영역 (Opportunity Gap)

IV. 결론 및 전략 제언 (Conclusion & Strategy)
1. 분석 요약 및 최종 결론
2. 향후 전략 방향 (Go-to-Market 전략 또는 핵심 액션 플랜)
"""

    industry_detailed_resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=industry_detailed_prompt,
        config=types.GenerateContentConfig(response_mime_type="text/plain"),
    )
    industry_detailed_report = (industry_detailed_resp.text or "").strip()

    filename_industry = f"Industry_Detail_{company_name}_{datetime.date.today()}.md"
    download_button("📥 산업군 상세 리포트(A4 1~2장) 다운로드", industry_detailed_report, filename_industry)
