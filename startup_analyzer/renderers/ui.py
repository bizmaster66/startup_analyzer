from typing import Optional

import streamlit as st


def configure_page():
    st.set_page_config(layout="wide", page_title="혁신의숲 Startup Analyzer & Report")


def render_page_header():
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


def render_sidebar():
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


def render_step(step: int):
    if step == 1:
        html_block = """
<div style="display:flex;gap:12px;margin-bottom:24px;">
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 1 · 정보 수집</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#F1F5F9;color:#475569;">
STEP 2 · JSON 분석</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#F1F5F9;color:#475569;">
STEP 3 · 결과 생성</div>
</div>
"""
    elif step == 2:
        html_block = """
<div style="display:flex;gap:12px;margin-bottom:24px;">
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 1 · 정보 수집</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 2 · JSON 분석</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#F1F5F9;color:#475569;">
STEP 3 · 결과 생성</div>
</div>
"""
    else:
        html_block = """
<div style="display:flex;gap:12px;margin-bottom:24px;">
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 1 · 정보 수집</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 2 · JSON 분석</div>
<div style="flex:1;padding:14px;text-align:center;border-radius:6px;
border:1px solid #CBD5E1;background:#00D2A8;color:white;font-weight:700;">
STEP 3 · 결과 생성</div>
</div>
"""
    st.write(html_block, unsafe_allow_html=True)


def tile(title: str, body: str):
    safe = str(body or "").replace("\n", "<br>")
    st.write(
        f"""
<div style="background:#E2E8F0;padding:10px 14px;border-radius:6px 6px 0 0;
border:1px solid #CBD5E1;font-weight:600;">{title}</div>""",
        unsafe_allow_html=True,
    )
    st.write(
        f"""
<div style="background:white;padding:16px;border-radius:0 0 6px 6px;
border:1px solid #CBD5E1;border-top:none;font-size:14px;line-height:1.6;">
{safe}</div>""",
        unsafe_allow_html=True,
    )
    st.write("<div style='height:14px;'></div>", unsafe_allow_html=True)


def render_api_key_error():
    st.error(
        "Gemini API Key가 설정되지 않았습니다.\n\n"
        "로컬 실행:\n"
        "  - 터미널에서 `export GEMINI_API_KEY=\"YOUR_KEY\"` 설정 후 실행\n\n"
        "Streamlit Cloud 배포:\n"
        "  - App settings → Secrets 에 아래처럼 추가\n"
        "    GEMINI_API_KEY=\"YOUR_KEY\"\n"
        "    GOOGLE_API_KEY=\"YOUR_KEY\" (선택)"
    )


def render_input_form() -> tuple[str, str, str, bool]:
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
    return company_name, ceo_name, raw_text, run
