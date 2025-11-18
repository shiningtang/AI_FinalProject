# app/query_api.py
import os
import json
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .vector_store import get_vector_store
from .config import get_settings
import datetime

load_dotenv()
settings = get_settings()

CHAT_MODEL = settings.chat_model
OPENAI_API_KEY = settings.openai_api_key

# init OpenAI client (new SDK preferred, fallback to legacy)
openai_client = None
use_new_sdk = False
try:
    from openai import OpenAI as OpenAIClient
    openai_client = OpenAIClient(api_key=OPENAI_API_KEY)
    use_new_sdk = True
except Exception:
    try:
        import openai as openai_legacy
        openai_legacy.api_key = OPENAI_API_KEY
        openai_client = openai_legacy
        use_new_sdk = False
    except Exception as e:
        raise RuntimeError("OpenAI SDK not available or OPENAI_API_KEY missing") from e

store = get_vector_store()
app = FastAPI()

class QueryIn(BaseModel):
    q: str
    top_k: int = 4

def _json_extract(text: str):
    """
    Try to find a JSON object/array inside a model reply.
    Returns parsed object or None.
    """
    # common pattern: code block ```json {...}``` or raw {...}
    # first try to find a top-level JSON substring with braces
    # greedy but reasonably safe approach: find first { and last } and try parse
    try:
        # remove surrounding backticks and leading text like "```json"
        stripped = re.sub(r"```(?:json)?\n?", "", text, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"\n?```$", "", stripped, flags=re.IGNORECASE).strip()
        # find first { and last } (works for objects)
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start:end+1]
            return json.loads(candidate)
    except Exception:
        pass
    # try to find array
    try:
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start:end+1]
            return json.loads(candidate)
    except Exception:
        pass
    # last resort: try to parse whole stripped text
    try:
        return json.loads(stripped)
    except Exception:
        return None
    
def _safe_serialize_value(v):
    """把任意 metadata 值轉成 JSON-safe 的字串表示"""
    if v is None:
        return ""
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    try:
        # 有時候是 Decimal / numpy types，先嘗試直接轉字串
        return str(v)
    except Exception:
        return ""

def build_prompt(question: str, retrieved: list[dict]) -> str:
    """
    Build prompt that provides each retrieved DB row as an explicit JSON-like block.
    Ensure metadata values are serialized to JSON-safe strings.
    """
    instruction = (
        "你是研究所申請小助手。**請僅根據下列每筆資料列出的欄位回答，不要憑空推測或加入資料庫外的資訊。**\n"
        "每筆資料都已以 JSON 格式呈現（欄位可能為空的寫為空字串）。\n"
        "請務必以 JSON 輸出，格式精確為：\n"
        "{\n"
        "  \"short_answer\": \"一句話回答（中文）\",\n"
        "  \"detail\": \"詳細中文說明（若可列出必要文件與具體截止日請直接寫出，若欄位為空則寫「無資料」）\",\n"
        "  \"sources_by_row\": [\n"
        "     {\"row_index\": 1, \"note\": \"(可選，簡短說明此 row 為何被用)\"},\n"
        "     ...\n"
        "  ]\n"
        "}\n\n"
        "如果某欄位在該 row 中有值，請直接引用該欄位值（例如 deadline: 2025-10-02 就在 detail 中寫出 '截止日：2025-10-02'）。\n"
        "若找不到必要文件或其他欄位請明說「無資料」。\n"
        "現在給你使用者問題與檢索到的 row（每筆 row 編號自 1 起）：\n\n"
    )
    docs = ""
    for i, r in enumerate(retrieved, start=1):
        meta = (r.get("metadata") or {}) if isinstance(r.get("metadata", {}), dict) else {}
        # build safe dict for JSON printing
        meta_for_prompt = {
            "school": _safe_serialize_value(meta.get("school", "")),
            "college": _safe_serialize_value(meta.get("college", "")),
            "department": _safe_serialize_value(meta.get("department", "")),
            "track": _safe_serialize_value(meta.get("track", "")),
            "deadline": _safe_serialize_value(meta.get("deadline", "")),
            "quota": _safe_serialize_value(meta.get("quota", "")),
            "assessment_weights": _safe_serialize_value(meta.get("assessment_weights", "")),
            "other_req": _safe_serialize_value(meta.get("other_req", "")),
            "docs_required": _safe_serialize_value(meta.get("docs_required", "")),
            "interview_required": _safe_serialize_value(meta.get("interview_required", "")),
            "written_exam_required": _safe_serialize_value(meta.get("written_exam_required", "")),
            "link_apply": _safe_serialize_value(meta.get("link_apply", "") or meta.get("url", "")),
            "note": _safe_serialize_value(meta.get("note", ""))
        }
        text_field = r.get("text", "") or ""
        # now json.dumps will not fail because meta_for_prompt values are strings
        docs += f"--- ROW {i} ---\n"
        docs += json.dumps(meta_for_prompt, ensure_ascii=False) + "\n"
        docs += "text:\n" + text_field + "\n\n"

    prompt = f"{instruction}使用者問題: {question}\n\n檢索到的 rows:\n{docs}\n請直接輸出 JSON（不要其他文字）。"
    return prompt

@app.post("/query")
async def query(q_in: QueryIn):
    q = q_in.q
    top_k = q_in.top_k
    # retrieval
    try:
        retrieved = store.query(q, top_k=top_k)
    except Exception as e:
        print("Retrieval error:", e)
        raise HTTPException(status_code=503, detail="Retrieval failed (DB/network).")

    # Build prompt with full rows (metadata + text)
    prompt = build_prompt(q, retrieved)

    # call LLM
    try:
        if use_new_sdk:
            # new SDK: chat completions under .chat.completions
            resp = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.0
            )
            answer_raw = resp.choices[0].message.content
        else:
            resp = openai_client.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.0
            )
            # legacy shape
            answer_raw = resp.choices[0].message["content"] if isinstance(resp.choices[0].message, dict) else resp.choices[0].message.content
    except Exception as e:
        print("OpenAI error:", e)
        raise HTTPException(status_code=502, detail="LLM generation failed")

    # try to extract JSON from answer
    parsed = _json_extract(answer_raw)
    if parsed is None:
        # parsing failed -> return raw answer but include full rows in sources so client can still inspect
        sources = []
        for i, r in enumerate(retrieved, start=1):
            sources.append({
                "row_index": i,
                "metadata": r.get("metadata"),
                "text": r.get("text", "")
            })
        return {
            "question": q,
            "answer_raw": answer_raw,
            "answer_struct": None,
            "sources": sources,
            "note": "LLM output JSON parsing failed; see answer_raw"
        }

    # parsed is a dict with short_answer/detail/sources_by_row ideally
    sources = []
    for i, r in enumerate(retrieved, start=1):
        sources.append({
            "row_index": i,
            "metadata": r.get("metadata"),
            "text": r.get("text", "")
        })

    return {
        "question": q,
        "answer_raw": answer_raw,
        "answer_struct": parsed,
        "sources": sources
    }
