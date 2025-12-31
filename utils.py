import json
import re
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from textwrap import wrap


# -------------------------------------------------------------
# JSON 추출 (중복 JSON 완전 대응)
# -------------------------------------------------------------
def extract_json_from_text(text: str):
    """
    모델 응답 텍스트에서 JSON을 추출한다.
    JSON이 여러 개 붙어서 나오는 경우 첫 번째 JSON만 사용.
    """
    if not text:
        raise ValueError("빈 응답입니다. JSON 추출 실패.")

    cleaned = (
        text.replace("```json", "")
            .replace("```", "")
            .strip()
    )

    matches = re.findall(r"\{[\s\S]*?\}", cleaned)

    if not matches:
        raise ValueError("JSON 객체를 찾지 못했습니다.")

    return json.loads(matches[0])


# -------------------------------------------------------------
# 산업 키워드 자동 생성
# -------------------------------------------------------------
def extract_industry_keywords(profile_data):
    raw = profile_data.get("industry_keywords", [])
    clean = [k for k in raw if k and "확인 불가" not in k]

    if clean:
        return clean

    # product feature 기반 자동 키워드 생성
    feat = profile_data.get("product_core_features", [])
    tokens = " ".join(feat).lower().split()

    auto_kw = []
    for t in tokens:
        if len(t) > 3:
            auto_kw.append(t)

    auto_kw = list(set(auto_kw))[:5]

    if not auto_kw:
        auto_kw = ["tech", "platform"]

    return auto_kw


# -------------------------------------------------------------
# PDF 생성
# -------------------------------------------------------------
def generate_pdf(profile, report_text, file_path="startup_analysis.pdf"):
    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4

    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Startup Analysis Report")
    y -= 40

    def draw_block(title, content, y):
        c.setFont("Helvetica-Bold", 13)
        c.drawString(50, y, title)
        y -= 20

        c.setFont("Helvetica", 11)
        wrapped = wrap(content, 95)

        for line in wrapped:
            c.drawString(50, y, line)
            y -= 14
            if y < 70:
                c.showPage()
                y = height - 50
        return y - 20

    # profile 출력
    for key, value in profile.items():
        y = draw_block(key, str(value), y)

    # industry report
    y = draw_block("Industry Report", report_text, y)

    c.save()
    return file_path
