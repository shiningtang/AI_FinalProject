# app/main.py
from __future__ import annotations
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, FlexSendMessage,
    PostbackEvent, QuickReply, QuickReplyButton, PostbackAction,
)

from .config import get_settings
from .data import load_dataframe, parse_query, filter_rows
from .flex import build_department_bubble

# 在檔案頂部 imports 區加：
import os
import requests
import threading
from dotenv import load_dotenv
load_dotenv()



settings = get_settings()
app = FastAPI(title="LINE Grad Admissions Bot")

line_bot_api = LineBotApi(settings.line_channel_access_token)
handler = WebhookHandler(settings.line_channel_secret)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/callback")  # for LINE verify
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

# ... 你既有的 imports 保留 ...
# 後面在 settings 取得或直接用 env
QUERY_API_URL = os.environ.get("QUERY_API_URL", "http://127.0.0.1:8000/query")
CLIENT_API_KEY = os.environ.get("API_KEY_FOR_CLIENT", "")

# --- helper: call your query API (sync, with configurable timeout) ---
def call_query_api(user_text: str, top_k: int = 3, timeout: int = 6):
    headers = {"Content-Type": "application/json"}
    if CLIENT_API_KEY:
        headers["x-api-key"] = CLIENT_API_KEY
    try:
        resp = requests.post(QUERY_API_URL, json={"q": user_text, "top_k": top_k}, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return {"error": "timeout"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

# --- helper: prepare a text answer from your API response (simple fallback) ---
def extract_answer_text(api_resp: dict) -> str:
    # adapt to your actual API shape (you used 'answer' or 'answer_raw' earlier)
    if not api_resp:
        return "抱歉，查無資料。"
    if api_resp.get("error"):
        return f"查詢失敗：{api_resp.get('error')}"
    # try common keys
    if "answer" in api_resp:
        return api_resp["answer"]
    if "answer_raw" in api_resp:
        return api_resp["answer_raw"]
    # if your response contains structured fields:
    if "answer_text" in api_resp:
        return api_resp["answer_text"]
    # fallback: compile from sources
    sources = api_resp.get("sources", [])
    if sources:
        lines = []
        for s in sources[:3]:
            meta = s.get("meta") or s.get("metadata") or s.get("metadata", {})
            title = f"{meta.get('school','')}-{meta.get('department','')}".strip("-")
            excerpt = s.get("excerpt") or s.get("text") or ""
            lines.append(f"{title}: {excerpt[:200].strip()}")
        return "我找到以下相關資料：\n" + "\n\n".join(lines)
    return "抱歉，找不到相關資料。"

# --- background worker to query and push result to user ---
def background_query_and_push(user_id: str, user_text: str, top_k: int = 3):
    # longer timeout for background retrieval
    api_resp = call_query_api(user_text, top_k=top_k, timeout=30)
    answer = extract_answer_text(api_resp)

    # chunk if too long for single LINE message
    max_len = 1800
    msgs = []
    for i in range(0, len(answer), max_len):
        msgs.append(TextSendMessage(text=answer[i:i+max_len]))

    try:
        # push_message requires user to have interacted with the bot (or be friend)
        line_bot_api.push_message(user_id, msgs)
    except Exception as e:
        print("Failed to push message to user:", e)

# ----------------- 修改 handle_message 的核心邏輯 ---------------
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    user_text = event.message.text.strip()
    # 先原有本地 dataframe 檢索邏輯（保留你目前的行為）
    df = load_dataframe()
    parsed = parse_query(user_text)

    hits_df = filter_rows(df, parsed["tokens"], parsed["month"], limit=None)
    total = len(hits_df)
    rows = hits_df.to_dict(orient="records")

    # 若在本地查得到（你的舊邏輯）就按原樣處理：
    if total == 0:
        # 沒命中本地資料：嘗試用你的 query API（短 timeout 6s）
        api_resp = call_query_api(user_text, top_k=3, timeout=6)
        if api_resp.get("error") == "timeout":
            # 立即告知使用者並在 background 把答案 push 回來
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="查詢時間較久，正在為您查詢，查詢結果會稍後發送。"))
            # spawn background thread to call and push
            user_id = event.source.user_id
            t = threading.Thread(target=background_query_and_push, args=(user_id, user_text, 3), daemon=True)
            t.start()
            return
        elif api_resp.get("error"):
            # 其他 error（例如 network/db）→ 回覆友善錯誤
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="系統查詢失敗，請稍後再試。"))
            print("Query API error:", api_resp.get("error"))
            return
        else:
            # 即時成功，直接回覆（reply_token 還有效）
            answer = extract_answer_text(api_resp)
            # chunk large answers
            max_len = 1800
            msgs = []
            for i in range(0, len(answer), max_len):
                msgs.append(TextSendMessage(text=answer[i:i+max_len]))
            try:
                line_bot_api.reply_message(event.reply_token, msgs)
            except Exception as e:
                print("Reply failed (maybe token expired):", e)
                # fallback: try push (if we have user id)
                try:
                    line_bot_api.push_message(event.source.user_id, msgs)
                except Exception as ex:
                    print("Push fallback failed:", ex)
            return

    # 若 local 有 hits 才走你既有 intent / bubble 邏輯（保留你原本的邏輯）
    # --- the rest of your original logic continues exactly as before ---
    intent = parsed.get("intent")
    if intent:
        field = _FIELD_MAP.get(intent)
        if field:
            if total == 1:
                r = rows[0]
                title = _title_for_row(r)
                val = str(r.get(field, "")).strip()
                if field == "portfolio_required":
                    v = val.lower()
                    val = "需要" if v in {"true","1","yes","y","需要"} else ("不需要" if val != "" else "—")
                text = _format_single_answer(_LABEL_MAP.get(field, field), val, title)
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=text))
                return
            elif 2 <= total <= 10:
                _ask_disambiguation(event, field, rows)
                return
            else:
                _send_need_narrow_message(event)
                return

    n = len(rows)
    if n > 10:
        _send_need_narrow_message(event)
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
