import json
from typing import Any, Dict

from google import genai
from google.genai import types


def _escape_inner_quotes_heuristic(text: str) -> str:
    out = []
    in_str = False
    esc = False
    i = 0
    n = len(text)

    def next_non_space(idx: int) -> str:
        j = idx
        while j < n and text[j].isspace():
            j += 1
        return text[j] if j < n else ""

    while i < n:
        ch = text[i]

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
                in_str = True
                out.append(ch)
            else:
                nxt = next_non_space(i + 1)
                if nxt in [",", "}", "]"]:
                    in_str = False
                    out.append(ch)
                else:
                    out.append('\\"')
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty response")

    cleaned = text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON block not found")

    raw = cleaned[start : end + 1]
    try:
        return json.loads(raw)
    except Exception:
        repaired = _escape_inner_quotes_heuristic(raw)
        return json.loads(repaired)


def repair_json_with_model(
    client: genai.Client,
    model_name: str,
    raw_text: str,
    schema_hint: str = "",
) -> str:
    fix_prompt = (
        "아래 출력은 JSON 형식이 깨져 있습니다. 내용을 최대한 동일하게 유지하되,\n"
        "반드시 표준 JSON으로만 수정해서 JSON만 출력하세요.\n\n"
        "[규칙]\n"
        "- 코드펜스, 설명 문장, 주석 금지\n"
        "- 문자열 내부 큰따옴표는 JSON 문법에 맞게 처리\n"
        "- 키 이름과 구조는 유지\n"
        "- JSON ONLY\n\n"
    )
    if schema_hint:
        fix_prompt += f"[스키마 참고]\n{schema_hint}\n\n"
    fix_prompt += f"[원본]\n{raw_text}\n"

    response = client.models.generate_content(
        model=model_name,
        contents=fix_prompt,
        config=types.GenerateContentConfig(response_mime_type="text/plain"),
    )
    return (response.text or "").strip()
