import re
from typing import Any, List


def normalize_text_list(values: Any, limit: int = 6) -> List[str]:
    if isinstance(values, list):
        raw_items = values
    elif values:
        raw_items = [values]
    else:
        raw_items = []

    output = []
    seen = set()
    for item in raw_items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def extract_keywords(profile: dict) -> List[str]:
    kws = [k for k in profile.get("industry_keywords", []) if "확인 불가" not in str(k)]
    if kws:
        return kws

    features = profile.get("product_core_features", [])
    text = " ".join([str(x) for x in features]) if isinstance(features, list) else str(features)
    auto = [token for token in text.lower().split() if len(token) > 3]
    return list(dict.fromkeys(auto))[:5] if auto else ["technology"]


def safe_filename(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9가-힣._-]+", "_", str(text or "").strip())
    return value.strip("_") or "report"


def clean_korean_label(text: Any, fallback: str = "") -> str:
    value = str(text or "").strip()
    if not value:
        return fallback

    value = re.sub(r"[一-龯ぁ-ゔゞァ-・ヽヾ゛゜ー]", "", value)
    value = re.sub(r"[^0-9A-Za-z가-힣/&().,+\- ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" -_/")
    return value or fallback
