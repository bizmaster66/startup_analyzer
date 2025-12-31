# Startup Analyzer Pro

## 1) 설치
```bash
pip install -r requirements.txt
```

## 2) API 키 설정 (필수)
이 앱은 Google Gemini API를 사용합니다. 아래 중 한 가지 방식으로 키를 설정하세요.

### 로컬 실행
```bash
export GEMINI_API_KEY="YOUR_KEY"
# 또는
export GOOGLE_API_KEY="YOUR_KEY"
```

### Streamlit Cloud 배포
- App settings → **Secrets**에 아래를 추가:
```toml
GEMINI_API_KEY="YOUR_KEY"
# (보험) 코드에 따라 아래도 함께 넣어도 됩니다.
GOOGLE_API_KEY="YOUR_KEY"
```

> ⚠️ `.env`, `.venv` 같은 민감/로컬 파일은 GitHub에 올리지 마세요.

## 3) 실행
```bash
streamlit run app.py
```
