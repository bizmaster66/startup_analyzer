import datetime

import streamlit as st

from startup_analyzer.renderers.bmc import (
    build_bmc_markdown,
    build_overview_report_markdown,
    render_bmc,
)
from startup_analyzer.renderers.ui import (
    configure_page,
    render_api_key_error,
    render_input_form,
    render_page_header,
    render_sidebar,
    render_step,
    tile,
)
from startup_analyzer.services.analysis import (
    build_client,
    gather_company_facts,
    generate_company_profile,
    get_gemini_api_key,
)
from startup_analyzer.services.bmc import build_bmc_and_diagram_data
from startup_analyzer.services.diagram_image import generate_bm_diagram_png
from startup_analyzer.utils.text import extract_keywords, normalize_text_list, safe_filename


def main():
    configure_page()
    render_page_header()
    render_sidebar()

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    company_name, ceo_name, raw_text, run = render_input_form()
    if run:
        if not company_name.strip():
            st.error("기업명을 입력해주세요.")
            return
        if not ceo_name.strip():
            st.error("대표자명을 입력해주세요.")
            return

        api_key = get_gemini_api_key()
        if not api_key:
            render_api_key_error()
            return

        client = build_client(api_key)

        render_step(1)
        facts = gather_company_facts(client, company_name, ceo_name, raw_text)

        render_step(2)
        try:
            profile = generate_company_profile(client, company_name, facts)
        except Exception as exc:
            st.error(f"기업 분석 JSON 생성에 실패했습니다: {exc}")
            return

        keywords = extract_keywords(profile)
        try:
            bmc_data = build_bmc_and_diagram_data(client, company_name, ceo_name, facts, profile, keywords)
        except Exception as exc:
            st.error(f"BMC 및 BM 다이어그램 데이터 생성에 실패했습니다: {exc}")
            return

        try:
            diagram_png = generate_bm_diagram_png(client, company_name, bmc_data)
        except Exception as exc:
            st.error(f"BM 다이어그램 이미지 생성에 실패했습니다: {exc}")
            return
        overview_report_md = build_overview_report_markdown(company_name, ceo_name, profile, keywords, bmc_data)
        bmc_md = build_bmc_markdown(bmc_data)
        st.session_state.analysis_result = {
            "company_name": company_name,
            "ceo_name": ceo_name,
            "profile": profile,
            "keywords": keywords,
            "bmc_data": bmc_data,
            "diagram_png": diagram_png,
            "overview_report_md": overview_report_md,
            "bmc_md": bmc_md,
        }

    result = st.session_state.analysis_result
    if not result:
        return

    company_name = result["company_name"]
    ceo_name = result["ceo_name"]
    profile = result["profile"]
    keywords = result["keywords"]
    bmc_data = result["bmc_data"]
    diagram_png = result["diagram_png"]
    overview_report_md = result["overview_report_md"]
    bmc_md = result["bmc_md"]

    render_step(3)
    st.markdown("## 기업 분석 결과")
    tile("문제 정의", profile.get("problem_definition", ""))
    tile("솔루션 및 제공 가치", profile.get("solution_value_prop", ""))
    tile("비즈니스 모델", profile.get("revenue_model_type", ""))
    tile("핵심 기능", "<br>".join(normalize_text_list(profile.get("product_core_features", []), limit=8)))
    tile("핵심 기술 · 경쟁력", profile.get("core_tech_moat", ""))
    tile("대표자 비전", profile.get("ceo_vision_summary", ""))
    tile("조직 · 운영 방식", profile.get("org_culture_biz_focus", ""))
    tile("최근 뉴스 요약", profile.get("recent_news_summary", ""))
    tile("산업 키워드", ", ".join(keywords))

    st.markdown("## Business Model Canvas")
    if bmc_data.get("bm_type"):
        tile("BM 유형", bmc_data.get("bm_type", ""))
    if bmc_data.get("bmc_summary"):
        tile("BMC 요약", bmc_data.get("bmc_summary", ""))
    render_bmc(bmc_data)

    st.markdown("## BM 다이어그램")
    st.image(diagram_png, use_container_width=True)

    st.markdown("## 다운로드")
    safe_name = safe_filename(company_name)
    today = datetime.date.today()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button(
            "기업개요 리포트 다운로드",
            data=overview_report_md,
            file_name=f"Company_Overview_{safe_name}_{today}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_b:
        st.download_button(
            "BMC 다운로드",
            data=bmc_md,
            file_name=f"BMC_{safe_name}_{today}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_c:
        st.download_button(
            "BM 다이어그램 다운로드",
            data=diagram_png,
            file_name=f"BM_Diagram_{safe_name}_{today}.png",
            mime="image/png",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
