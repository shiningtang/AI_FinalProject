from __future__ import annotations
import json
import re
import os
import threading
import datetime
import pprint
from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from .config import get_settings
from .vector_store import get_vector_store

load_dotenv()

# ==================== Settings & Initialization ====================
settings = get_settings()
app = FastAPI(title="LINE Grad Admissions Bot with Query API (RAG-first)")

# LINE Bot setup
line_bot_api = LineBotApi(settings.line_channel_access_token)
handler = WebhookHandler(settings.line_channel_secret)

# OpenAI setup for query API (supports new & legacy SDKs)
CHAT_MODEL = getattr(settings, "chat_model", os.getenv("CHAT_MODEL", "gpt-4o-mini"))
OPENAI_API_KEY = getattr(settings, "openai_api_key", os.getenv("OPENAI_API_KEY", ""))

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
    except Exception:
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
    if not text or not isinstance(text, str):
        return None
    try:
        # Remove code fences and leading "```json"
        stripped = re.sub(r"```(?:json)?\n?", "", text, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"\n?```$", "", stripped, flags=re.IGNORECASE).strip()

        # Try to find JSON object
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start:end + 1]
            return json.loads(candidate)
    except Exception:
        pass

    # Try to find JSON array
    try:
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start:end + 1]
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
    Instruct the LLM to output JSON with:
      - short_answer: 一句話回答
      - detail: 更詳細說明
      - used_row_indexes: LLM 覺得最相關的 row index (array of ints, 1-based)
    NOTE: We will use these indexes to fetch full DB metadata on the server side.
    """
    instruction = (
        "你是研究所申請小助手。請僅根據下列每筆資料列出的欄位回答，不要憑空推測或加入資料庫外的資訊。\n"
        "每筆資料今天已以 JSON 格式呈現(欄位可能為空的寫為空字串)。\n"
        "請務必以 JSON 輸出,格式精確為:\n"
        "{\n"
        "  \"short_answer\": \"一句話回答(中文)\",\n"
        "  \"detail\": \"詳細中文說明(若可列出必要文件與具體截止日請直接寫出,若欄位為空則寫「無資料」)\",\n        \"used_row_indexes\": [1, 2]  # 請回傳你使用的 row 的 1-based index\n"
        "}\n\n"
        "重要：你只要回傳上面那個 JSON （或包在 code fence 內也可），不要輸出其他多餘文字。\n"
        "如果你要指出某一筆 row 的原因，可把索引放到 used_row_indexes 中。\n\n"
        "現在給你使用者問題與檢索到的 rows（每筆 row 編號自 1 起）：\n\n"
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
    """從 API response 中提取答案文字（避免 short/detail 都是 '無資料' 時重覆回覆）。"""
    if not api_resp:
        return "抱歉, 查無資料。"

    if api_resp.get("error"):
        return f"查詢失敗: {api_resp.get('error')}"

    # 優先結構化
    if "answer_struct" in api_resp and api_resp["answer_struct"]:
        struct = api_resp["answer_struct"]
        short = (struct.get("short_answer") or "").strip()
        detail = (struct.get("detail") or "").strip()

        # 若 short/detail 都是空或 '無資料' -> 回一則「查無資料」
        if (not short or short == "無資料") and (not detail or detail == "無資料"):
            return "抱歉，查無相關資料。"
        # 若只有 short 有內容
        if short and (not detail or detail == "無資料"):
            return short
        # 若只有 detail 有內容或都有 -> short + detail
        if short and detail:
            return f"{short}\n\n{detail}"
        return detail or short or "抱歉，查無相關資料。"

    # fallback: raw
    if "answer_raw" in api_resp and api_resp["answer_raw"]:
        return api_resp["answer_raw"]

    if "answer" in api_resp and api_resp["answer"]:
        return api_resp["answer"]

    # 最後整理 sources 摘要
    sources = api_resp.get("sources", [])
    if sources:
        lines = []
        for s in sources[:3]:
            meta = s.get("metadata") or {}
            title = f"{meta.get('school','')}-{meta.get('department','')}".strip("-")
            excerpt = (s.get("text", "") or "")[:200].strip()
            lines.append(f"{title}: {excerpt}")
        return "我找到以下相關資料:\n\n" + "\n\n".join(lines)

    return "抱歉, 找不到相關資料。"


# ==================== Query API Endpoint (internal) ====================
@app.post("/query")
async def query_endpoint(q_in: QueryIn):
    """Internal query API endpoint used by LINE and other clients."""
    if store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    if openai_client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized")

    q = q_in.q
    top_k = q_in.top_k

    try:
        retrieved = store.query(q, top_k=top_k)
    except Exception as e:
        print("Retrieval error:", e)
        raise HTTPException(status_code=503, detail="Retrieval failed (DB/network).")

    prompt = build_prompt(q, retrieved)

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

    parsed = _json_extract(answer_raw)

    # default: take top retrieved rows if LLM didn't provide indexes
    used_indexes = []
    if isinstance(parsed, dict):
        # preferred key used_row_indexes (array of ints)
        if "used_row_indexes" in parsed and isinstance(parsed["used_row_indexes"], (list, tuple)):
            for v in parsed["used_row_indexes"]:
                try:
                    used_indexes.append(int(v))
                except Exception:
                    pass
        # fallback: some outputs might return sources_by_row = [{"row_index": 1,...}, ...]
        elif "sources_by_row" in parsed and isinstance(parsed["sources_by_row"], (list, tuple)):
            for item in parsed["sources_by_row"]:
                if isinstance(item, dict) and "row_index" in item:
                    try:
                        used_indexes.append(int(item["row_index"]))
                    except Exception:
                        pass

    # ensure indexes are unique, valid, and 1-based -> convert to 0-based for list access
    clean_idxs = []
    for idx in used_indexes:
        if isinstance(idx, int) and 1 <= idx <= len(retrieved):
            if idx not in clean_idxs:
                clean_idxs.append(idx)
    if not clean_idxs:
        # fallback: use all returned retrieved (or top_k)
        clean_idxs = list(range(1, min(len(retrieved), top_k) + 1))

    # prepare matched_rows from DB metadata and text (safe-serialized)
    matched_rows = []
    for idx in clean_idxs:
        r = retrieved[idx - 1]  # retrieved is 0-based
        meta_raw = r.get("metadata") or {}
        safe_meta = {
            "school": _safe_serialize_value(meta_raw.get("school", "")),
            "college": _safe_serialize_value(meta_raw.get("college", "")),
            "department": _safe_serialize_value(meta_raw.get("department", "")),
            "track": _safe_serialize_value(meta_raw.get("track", "")),
            "deadline": _safe_serialize_value(meta_raw.get("deadline", "")),
            "quota": _safe_serialize_value(meta_raw.get("quota", "")),
            "assessment_weights": _safe_serialize_value(meta_raw.get("assessment_weights", "")),
            "other_req": _safe_serialize_value(meta_raw.get("other_req", "")),
            "docs_required": _safe_serialize_value(meta_raw.get("docs_required", "")),
            "interview_required": _safe_serialize_value(meta_raw.get("interview_required", "")),
            "written_exam_required": _safe_serialize_value(meta_raw.get("written_exam_required", "")),
            "link_apply": _safe_serialize_value(meta_raw.get("link_apply", "") or meta_raw.get("url", "")),
            "note": _safe_serialize_value(meta_raw.get("note", "")),
        }
        matched_rows.append({
            "row_index": idx,
            "metadata": safe_meta,
            "text": r.get("text", "") or ""
        })

    # Build answer_struct for returning to client (LLM short answer + DB matched rows)
    answer_struct = {
        "short_answer": (parsed.get("short_answer", "") if isinstance(parsed, dict) else "") if parsed else "",
        "detail": (parsed.get("detail", "") if isinstance(parsed, dict) else "") if parsed else "",
        "matched_rows": matched_rows,
        "llm_parsed": parsed  # optional: include raw parsed JSON from LLM for debugging
    }

    return {
        "question": q,
        "answer_raw": answer_raw,
        "answer_struct": answer_struct,
        "sources": matched_rows
    }



# ==================== Internal Query Helper (used by LINE) ====================
def call_internal_query(user_text: str, top_k: int = 3):
    """Call internal query logic directly (no HTTP). Returns dict structured like /query response."""
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

        used_indexes = []
        if isinstance(parsed, dict):
            if "used_row_indexes" in parsed and isinstance(parsed["used_row_indexes"], (list, tuple)):
                for v in parsed["used_row_indexes"]:
                    try:
                        used_indexes.append(int(v))
                    except Exception:
                        pass
            elif "sources_by_row" in parsed and isinstance(parsed["sources_by_row"], (list, tuple)):
                for item in parsed["sources_by_row"]:
                    if isinstance(item, dict) and "row_index" in item:
                        try:
                            used_indexes.append(int(item["row_index"]))
                        except Exception:
                            pass

        clean_idxs = []
        for idx in used_indexes:
            if isinstance(idx, int) and 1 <= idx <= len(retrieved):
                if idx not in clean_idxs:
                    clean_idxs.append(idx)
        if not clean_idxs:
            clean_idxs = list(range(1, min(len(retrieved), top_k) + 1))

        matched_rows = []
        for idx in clean_idxs:
            r = retrieved[idx - 1]
            meta_raw = r.get("metadata") or {}
            safe_meta = {
                "school": _safe_serialize_value(meta_raw.get("school", "")),
                "college": _safe_serialize_value(meta_raw.get("college", "")),
                "department": _safe_serialize_value(meta_raw.get("department", "")),
                "track": _safe_serialize_value(meta_raw.get("track", "")),
                "deadline": _safe_serialize_value(meta_raw.get("deadline", "")),
                "quota": _safe_serialize_value(meta_raw.get("quota", "")),
                "assessment_weights": _safe_serialize_value(meta_raw.get("assessment_weights", "")),
                "other_req": _safe_serialize_value(meta_raw.get("other_req", "")),
                "docs_required": _safe_serialize_value(meta_raw.get("docs_required", "")),
                "interview_required": _safe_serialize_value(meta_raw.get("interview_required", "")),
                "written_exam_required": _safe_serialize_value(meta_raw.get("written_exam_required", "")),
                "link_apply": _safe_serialize_value(meta_raw.get("link_apply", "") or meta_raw.get("url", "")),
                "note": _safe_serialize_value(meta_raw.get("note", "")),
            }
            matched_rows.append({"row_index": idx, "metadata": safe_meta, "text": r.get("text", "") or ""})

        return {
            "answer_raw": answer_raw,
            "answer_struct": {
                "short_answer": (parsed.get("short_answer", "") if isinstance(parsed, dict) else "") if parsed else "",
                "detail": (parsed.get("detail", "") if isinstance(parsed, dict) else "") if parsed else "",
                "matched_rows": matched_rows,
                "llm_parsed": parsed
            },
            "sources": matched_rows
        }

    except Exception as e:
        print(f"Internal query error: {e}")
        return {"error": str(e)}



# ==================== Background Query & Push (kept for push use) ====================
def background_query_and_push(user_id: str, user_text: str, top_k: int = 3):
    """Background worker to query and push result to user"""
    api_resp = call_internal_query(user_text, top_k=top_k)
    answer = extract_answer_text(api_resp)

    # Chunk if too long for single LINE message
    max_len = 1800
    msgs = []
    for i in range(0, len(answer), max_len):
        msgs.append(TextSendMessage(text=answer[i:i + max_len]))

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


# ==================== LINE Bot Message Handler (RAG-first) ====================
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    user_text = event.message.text.strip()

    try:
        api_resp = call_internal_query(user_text, top_k=3)
    except Exception as e:
        print("call_internal_query threw:", e)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="系統查詢失敗，請稍後再試。")
        )
        return

    # DEBUG: 印出完整回傳，方便排查 retrieval / LLM 輸出
    print("=== DEBUG: api_resp ===")
    pprint.pprint(api_resp)
    print("=== END DEBUG ===")

    if api_resp.get("error"):
        print("Query error:", api_resp.get("error"))
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="系統查詢失敗，請稍後再試（內部錯誤）。")
        )
        return

    # 1. 取出 LLM 的簡短/詳細回答
    answer_text = extract_answer_text(api_resp)

    # 2. 將 retrieval 的 sources（metadata）整理成你要的欄位（只保留可讀格式）
    sources = api_resp.get("sources", [])
    if sources:
        # meta_list = [] # <-- 不再需要儲存完整的 meta_list
        # 將標題放在可讀行列表的開頭
        readable_lines = ["\n--- 相關科系資料---"] 
        
        for s in sources:
            md = s.get("metadata") or {}
            # 使用 _safe_serialize_value（檔案中已有此函式），確保日期等型別安全
            meta_item = {
                "school": _safe_serialize_value(md.get("school", "")),
                "college": _safe_serialize_value(md.get("college", "")),
                "department": _safe_serialize_value(md.get("department", "")),
                "track": _safe_serialize_value(md.get("track", "")),
                "deadline": _safe_serialize_value(md.get("deadline", "")),
                "quota": _safe_serialize_value(md.get("quota", "")),
                "assessment_weights": _safe_serialize_value(md.get("assessment_weights", "")),
                "other_req": _safe_serialize_value(md.get("other_req", "")),
                "docs_required": _safe_serialize_value(md.get("docs_required", "")),
                "interview_required": _safe_serialize_value(md.get("interview_required", "")),
                "written_exam_required": _safe_serialize_value(md.get("written_exam_required", "")),
                "link_apply": _safe_serialize_value(md.get("link_apply", "") or md.get("url", "")),
                "note": _safe_serialize_value(md.get("note", ""))
            }
            # meta_list.append(meta_item) # <-- 移除這行，不再儲存 JSON 列表

            # 可讀格式（簡短）
            readable_lines.append(
                f"{meta_item['school']} | {meta_item['college']} | {meta_item['department']} | {meta_item['track']}"
            )
            readable_lines.append(f"截止: {meta_item['deadline']}  名額: {meta_item['quota']}")
            readable_lines.append(f"書審: {meta_item['docs_required']}")
            readable_lines.append(f"面試: {meta_item['interview_required']}  筆試: {meta_item['written_exam_required']}")
            readable_lines.append(f"連結: {meta_item['link_apply']}")
            readable_lines.append(f"備註: {meta_item['note']}")
            readable_lines.append("")  # 空行分隔
        
        # 3. 將 LLM 回答和可讀格式合併，不再包含 JSON 區塊
        answer_text = f"{answer_text}\n\n" + "\n".join(readable_lines)

    # 4. 清理多餘空行
    answer_text = "\n".join([ln.rstrip() for ln in answer_text.splitlines() if ln.strip() != ""])

    # 5. 斷段回覆（Line 長度限制）
    max_len = 1800
    msgs = []
    for i in range(0, len(answer_text), max_len):
        msgs.append(TextSendMessage(text=answer_text[i:i+max_len]))

    try:
        line_bot_api.reply_message(event.reply_token, msgs)
    except Exception as e:
        print("Reply failed:", e)
        # fallback push
        try:
            line_bot_api.push_message(event.source.user_id, msgs)
        except Exception as ex:
            print("Push fallback failed:", ex)