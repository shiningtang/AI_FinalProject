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

# --------- helpers for short answers / UI ----------
def _tidy_value(val: str) -> str:
    """
    保留原始的 1. 2. 3. 編號與換行；僅做最小清理：
    - 統一換行為 \n
    - 刪除多餘空白行
    - 去右側空白
    """
    v = (val or "").strip()
    if not v or v.lower() in {"nan", "none", "null"}:
        return "—"
    v = v.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in v.split("\n")]
    cleaned, blank = [], False
    for ln in lines:
        if ln.strip() == "":
            if not blank:
                cleaned.append("")
            blank = True
        else:
            cleaned.append(ln)
            blank = False
    return "\n".join(cleaned).strip()

def _title_for_row(r: dict) -> str:
    """學校 系名 分組（缺的就略過）"""
    parts = [
        str(r.get("school", "")).strip(),
        (str(r.get("department", "")) or str(r.get("program", ""))).strip(),
        str(r.get("track", "")).strip()
    ]
    return " ".join([p for p in parts if p])

def _format_single_answer(label: str, value: str, title: str = "") -> str:
    """
    書審｜國立政治大學 資訊科學系（智慧計算組）
    1. ...
    2. ...
    """
    v = _tidy_value(value)
    head = f"{label}"
    if title:
        head += f"｜{title}"
    return f"{head}\n{v}"

def _ask_disambiguation(event: MessageEvent, field: str, rows: list[dict]) -> None:
    """用 Quick Reply + Postback 讓使用者選擇正確的一筆（最多 10 個）"""
    items = []
    for r in rows[:10]:
        title = _title_for_row(r)
        data = json.dumps({"act": "pick",
                           "f": field,
                           "s": r.get("school","").strip(),
                           "d": (r.get("department") or r.get("program") or "").strip(),
                           "t": r.get("track","").strip()}, ensure_ascii=False)
        items.append(QuickReplyButton(
            action=PostbackAction(label=title[:20], data=data, display_text=title)
        ))
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="請問您找的是下列哪一個呢？（點一下選擇）", quick_reply=QuickReply(items=items))
    )

def _send_need_narrow_message(event: MessageEvent) -> None:
    tips = (
        "符合的結果超過 10 筆，請輸入更完整的關鍵字縮小範圍唷～\n"
        "• 加學校全名或簡稱：例「政大 資科」\n"
        "• 指定分組/學程：例「政大 資科 一般組」\n"
        "• 查欄位請帶關鍵詞：例「政大 資科 截止」「政大 資科 書審」\n"
        "• 也可加月份：例「政大 資科 10月 截止」"
    )
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=tips))

# intent -> 欄位/標籤對照
_FIELD_MAP = {
    "deadline": "deadline",
    "quota": "quota",
    "assessment_weights": "assessment_weights",
    "other_req": "other_req",
    "docs_required": "docs_required",
    "interview_required": "interview_required",
    "written_exam_required": "written_exam_required",
    # 後續可擴充
    "english": "english_required",
    "english_required": "english_required",
    "gpa": "gpa_min",
    "portfolio": "portfolio_required",
    "portfolio_required": "portfolio_required",
}
_LABEL_MAP = {
    "deadline": "截止", "quota": "名額",
    "assessment_weights": "審查資料", "other_req": "特殊要求",
    "docs_required": "書審", "interview_required": "面試",
    "written_exam_required": "筆試",
    "english_required": "英文", "gpa_min": "GPA", "portfolio_required": "作品集",
}

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent):
    user_text = event.message.text.strip()
    df = load_dataframe()
    parsed = parse_query(user_text)

    # 不截斷，先拿到全部命中數
    hits_df = filter_rows(df, parsed["tokens"], parsed["month"], limit=None)
    total = len(hits_df)
    rows = hits_df.to_dict(orient="records")

    if total == 0:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="找不到符合的結果😭\n試試「政大 資科」或「政大 10月 截止」吧～")
        )
        return

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

    # 沒意圖：這裡加入 >10 的處理
    n = len(rows)
    if n > 10:
        _send_need_narrow_message(event)
        return

    # 1～10 筆 → 全部列成 bubble / carousel
    bubbles = []
    for r in rows:  # 注意：不再切片 [:10]，因為 n 已保證 <= 10
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


@handler.add(PostbackEvent)
def handle_postback(event: PostbackEvent):
    try:
        data = json.loads(event.postback.data)
    except Exception:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="資料格式不正確，請再試一次。"))
        return
    if data.get("act") != "pick":
        return

    field  = data.get("f", "")
    s = data.get("s", ""); d = data.get("d", ""); t = data.get("t", "")

    df = load_dataframe()
    hit = df[
        (df["school"].str.strip() == s.strip()) &
        (df["department"].str.strip() == d.strip()) &
        (df["track"].str.strip() == t.strip())
    ]
    if hit.empty:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="找不到那一筆資料，請再試一次。"))
        return

    row = hit.iloc[0].to_dict()
    title = _title_for_row(row)
    value = str(row.get(field, "")).strip()
    if field == "portfolio_required":
        v = value.lower()
        value = "需要" if v in {"true","1","yes","y","需要"} else ("不需要" if value != "" else "—")

    text = _format_single_answer(_LABEL_MAP.get(field, field), value, title)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=text))
