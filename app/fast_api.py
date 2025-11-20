# app/main.py - Merged LINE Bot + Query API
from __future__ import annotations
import json
import re
import os
import requests
import threading
import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, FlexSendMessage,
    PostbackEvent, QuickReply, QuickReplyButton, PostbackAction,
)

from .config import get_settings
from .data import load_dataframe, parse_query, filter_rows
from .flex import build_department_bubble
from .vector_store import get_vector_store

load_dotenv()

# ==================== Settings & Initialization ====================
settings = get_settings()
app = FastAPI(title="LINE Grad Admissions Bot with Query API")

# LINE Bot setup
line_bot_api = LineBotApi(settings.line_channel_access_token)
handler = WebhookHandler(settings.line_channel_secret)

# OpenAI setup for query API
CHAT_MODEL = settings.chat_model
OPENAI_API_KEY = settings.openai_api_key

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
        print("Warning: OpenAI SDK not available. Query API will not work.")
        openai_client = None

# Vector store setup
try:
    store = get_vector_store()
except Exception as e:
    print(f"Warning: Vector store initialization failed: {e}")
    store = None


# ==================== Query API Models & Helpers ====================
class QueryIn(BaseModel):
    q: str
    top_k: int = 4


def _json_extract(text: str):
    """
    Try to find a JSON object/array inside a model reply.
    Returns parsed object or None.
    """
    try:
        # Remove surrounding backticks and leading text like "```json"
        stripped = re.sub(r"```(?:json)?\n?", "", text, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"\n?```$", "", stripped, flags=re.IGNORECASE).strip()
        
        # Try to find JSON object
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start:end+1]
            return json.loads(candidate)
    except Exception:
        pass
    
    # Try to find JSON array
    try:
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start:end+1]
            return json.loads(candidate)
    except Exception:
        pass
    
    # Last resort: try to parse whole stripped text
    try:
        return json.loads(stripped)
    except Exception:
        return None


def _safe_serialize_value(v):
    """將任意 metadata 值轉成 JSON-safe 的字串表示"""
    if v is None:
        return ""
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    try:
        return str(v)
    except Exception:
        return ""


def build_prompt(question: str, retrieved: list[dict]) -> str:
    """
    Build prompt that provides each retrieved DB row as an explicit JSON-like block.
    """
    instruction = (
        "你是研究所申請小助手。**請僅根據下列每筆資料列出的欄位回答,不要憑空推測或加入資料庫外的資訊。**\n"
        "每筆資料都已以 JSON 格式呈現(欄位可能為空的寫為空字串)。\n"
        "請務必以 JSON 輸出,格式精確為:\n"
        "{\n"
        "  \"short_answer\": \"一句話回答(中文)\",\n"
        "  \"detail\": \"詳細中文說明(若可列出必要文件與具體截止日請直接寫出,若欄位為空則寫「無資料」)\",\n"
        "  \"sources_by_row\": [\n"
        "     {\"row_index\": 1, \"note\": \"(可選,簡短說明此 row 為何被用)\"},\n"
        "     ...\n"
        "  ]\n"
        "}\n\n"
        "如果某欄位在該 row 中有值,請直接引用該欄位值(例如 deadline: 2025-10-02 就在 detail 中寫出 '截止日:2025-10-02')。\n"
        "若找不到必要文件或其他欄位請明說「無資料」。\n"
        "現在給你使用者問題與檢索到的 row(每筆 row 編號自 1 起):\n\n"
    )
    
    docs = ""
    for i, r in enumerate(retrieved, start=1):
        meta = (r.get("metadata") or {}) if isinstance(r.get("metadata", {}), dict) else {}
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
        
        docs += f"--- ROW {i} ---\n"
        docs += json.dumps(meta_for_prompt, ensure_ascii=False) + "\n"
        docs += "text:\n" + text_field + "\n\n"

    prompt = f"{instruction}使用者問題: {question}\n\n檢索到的 rows:\n{docs}\n請直接輸出 JSON(不要其他文字)。"
    return prompt


def extract_answer_text(api_resp: dict) -> str:
    """從 API response 中提取答案文字"""
    if not api_resp:
        return "抱歉,查無資料。"
    
    if api_resp.get("error"):
        return f"查詢失敗:{api_resp.get('error')}"
    
    # Check for structured answer
    if "answer_struct" in api_resp and api_resp["answer_struct"]:
        struct = api_resp["answer_struct"]
        short = struct.get("short_answer", "")
        detail = struct.get("detail", "")
        return f"{short}\n\n{detail}".strip()
    
    # Fallback to raw answer
    if "answer_raw" in api_resp:
        return api_resp["answer_raw"]
    
    # Try other common keys
    if "answer" in api_resp:
        return api_resp["answer"]
    
    # Last resort: compile from sources
    sources = api_resp.get("sources", [])
    if sources:
        lines = []
        for s in sources[:3]:
            meta = s.get("metadata") or {}
            title = f"{meta.get('school','')}-{meta.get('department','')}".strip("-")
            excerpt = s.get("text", "")[:200].strip()
            lines.append(f"{title}: {excerpt}")
        return "我找到以下相關資料:\n" + "\n\n".join(lines)
    
    return "抱歉,找不到相關資料。"


# ==================== Query API Endpoint ====================
@app.post("/query")
async def query_endpoint(q_in: QueryIn):
    """Internal query API endpoint"""
    if store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    if openai_client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized")
    
    q = q_in.q
    top_k = q_in.top_k
    
    # Retrieval
    try:
        retrieved = store.query(q, top_k=top_k)
    except Exception as e:
        print("Retrieval error:", e)
        raise HTTPException(status_code=503, detail="Retrieval failed (DB/network).")

    # Build prompt
    prompt = build_prompt(q, retrieved)

    # Call LLM
    try:
        if use_new_sdk:
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
            answer_raw = resp.choices[0].message["content"] if isinstance(resp.choices[0].message, dict) else resp.choices[0].message.content
    except Exception as e:
        print("OpenAI error:", e)
        raise HTTPException(status_code=502, detail="LLM generation failed")

    # Try to extract JSON from answer
    parsed = _json_extract(answer_raw)
    
    sources = []
    for i, r in enumerate(retrieved, start=1):
        sources.append({
            "row_index": i,
            "metadata": r.get("metadata"),
            "text": r.get("text", "")
        })

    if parsed is None:
        return {
            "question": q,
            "answer_raw": answer_raw,
            "answer_struct": None,
            "sources": sources,
            "note": "LLM output JSON parsing failed; see answer_raw"
        }

    return {
        "question": q,
        "answer_raw": answer_raw,
        "answer_struct": parsed,
        "sources": sources
    }


# ==================== Internal Query Helper ====================
def call_internal_query(user_text: str, top_k: int = 3):
    """Call internal query logic directly (no HTTP)"""
    if store is None or openai_client is None:
        return {"error": "Service not initialized"}
    
    try:
        retrieved = store.query(user_text, top_k=top_k)
        prompt = build_prompt(user_text, retrieved)
        
        if use_new_sdk:
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
            answer_raw = resp.choices[0].message["content"] if isinstance(resp.choices[0].message, dict) else resp.choices[0].message.content
        
        parsed = _json_extract(answer_raw)
        sources = [{"row_index": i+1, "metadata": r.get("metadata"), "text": r.get("text", "")} 
                   for i, r in enumerate(retrieved)]
        
        return {
            "answer_raw": answer_raw,
            "answer_struct": parsed,
            "sources": sources
        }
    except Exception as e:
        print(f"Internal query error: {e}")
        return {"error": str(e)}


# ==================== Background Query & Push ====================
def background_query_and_push(user_id: str, user_text: str, top_k: int = 3):
    """Background worker to query and push result to user"""
    api_resp = call_internal_query(user_text, top_k=top_k)
    answer = extract_answer_text(api_resp)

    # Chunk if too long for single LINE message
    max_len = 1800
    msgs = []
    for i in range(0, len(answer), max_len):
        msgs.append(TextSendMessage(text=answer[i:i+max_len]))

    try:
        line_bot_api.push_message(user_id, msgs)
    except Exception as e:
        print("Failed to push message to user:", e)


# ==================== LINE Bot Health & Callback ====================
@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/callback")
def callback_verify():
    return {"ok": True}


@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body_text = (await request.body()).decode("utf-8")
    try:
        handler.handle(body_text, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return JSONResponse({"status": "ok"})


# ==================== LINE Bot Message Handler ====================
# Note: You'll need to define these helper functions based on your original code:
# _FIELD_MAP, _LABEL_MAP, _title_for_row, _format_single_answer, 
# _ask_disambiguation, _send_need_narrow_message

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    user_text = event.message.text.strip()
    
    # 先原有本地 dataframe 檢索邏輯(保留你目前的行為)
    df = load_dataframe()
    parsed = parse_query(user_text)
    hits_df = filter_rows(df, parsed["tokens"], parsed["month"], limit=None)
    total = len(hits_df)
    rows = hits_df.to_dict(orient="records")

    # 若在本地查得到(你的舊邏輯)就按原樣處理:
    if total == 0:
        # 沒命中本地資料:嘗試用內部 query API
        api_resp = call_internal_query(user_text, top_k=3)
        
        if api_resp.get("error"):
            # 查詢失敗 → 回覆友善錯誤
            line_bot_api.reply_message(
                event.reply_token, 
                TextSendMessage(text="系統查詢失敗,請稍後再試。")
            )
            print("Query error:", api_resp.get("error"))
            return
        
        # 即時成功,直接回覆
        answer = extract_answer_text(api_resp)
        
        # Chunk large answers
        max_len = 1800
        msgs = []
        for i in range(0, len(answer), max_len):
            msgs.append(TextSendMessage(text=answer[i:i+max_len]))
        
        try:
            line_bot_api.reply_message(event.reply_token, msgs)
        except Exception as e:
            print("Reply failed:", e)
            # Fallback: try push
            try:
                line_bot_api.push_message(event.source.user_id, msgs)
            except Exception as ex:
                print("Push fallback failed:", ex)
        return

    # 若 local 有 hits 才走你既有 intent / bubble 邏輯(保留你原本的邏輯)
    # You need to implement these parts based on your original code:
    # - _FIELD_MAP, _LABEL_MAP
    # - _title_for_row
    # - _format_single_answer
    # - _ask_disambiguation
    # - _send_need_narrow_message
    
    # Example structure (you'll need to fill in the details):
    intent = parsed.get("intent")
    if intent:
        # Handle intent-based queries
        # [Your original intent handling logic here]
        pass
    
    # Handle multiple results with bubbles
    n = len(rows)
    if n > 10:
        # _send_need_narrow_message(event)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="找到太多結果,請提供更具體的條件。")
        )
        return

    bubbles = []
    for r in rows:
        try:
            bubbles.append(build_department_bubble(r))
        except Exception as e:
            print("skip bad row:", e)

    if not bubbles:
        print("No valid bubbles to send.")
        return

    if len(bubbles) == 1:
        msg = FlexSendMessage(alt_text="系所資訊", contents=bubbles[0])
    else:
        msg = FlexSendMessage(alt_text="查詢結果", contents={"type": "carousel", "contents": bubbles})
    
    line_bot_api.reply_message(event.reply_token, messages=[msg])