# app/flex.py
import json, os, re
from urllib.parse import quote_plus, urlsplit, urlunsplit, parse_qsl, quote

# 一次載入樣板
_TEMPLATE_PATH = "flex_templates/department_card.json"
if not os.path.exists(_TEMPLATE_PATH):
    raise FileNotFoundError(f"Flex template not found: {_TEMPLATE_PATH}")
with open(_TEMPLATE_PATH, "r", encoding="utf-8") as f:
    TEMPLATE = json.load(f)

def _boolish(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "y", "yes", "需要"}

def _val(row: dict, key: str, default: str = "—") -> str:
    s = str(row.get(key, "")).strip()
    return s if s else default

def _sanitize_http_url(u: str) -> str:
    """將 http(s) URL 重新組裝並 percent-encode path/query；若失敗回傳空字串。"""
    try:
        u = u.strip()
        if not (u.startswith("http://") or u.startswith("https://")):
            return ""
        parts = urlsplit(u)
        # 對 path、query 做編碼：把逗號、空白、中文…都編碼掉
        path  = quote(parts.path, safe="/-._~")  # 不保留逗號，所以 , 會變 %2C
        query_pairs = parse_qsl(parts.query, keep_blank_values=True)
        query = "&".join(f"{quote(k, safe='-._~')}={quote(v, safe='-._~')}" for k, v in query_pairs)
        rebuilt = urlunsplit((parts.scheme, parts.netloc, path, query, parts.fragment))
        # 最後再做一次很鬆的檢查
        if re.match(r"^https?://[^\s]+$", rebuilt):
            return rebuilt
    except Exception:
        pass
    return ""

def _safe_uri(row: dict) -> str:
    # 優先取 contact_url / link_apply
    raw = (row.get("contact_url") or row.get("link_apply") or "").strip()
    fixed = _sanitize_http_url(raw)
    if fixed:
        return fixed
    # 兜 Google 搜尋（一定合法）
    q = " ".join([row.get("school",""), row.get("college",""), row.get("department","")]).strip() or "研究所 報名 網頁"
    return f"https://www.google.com/search?q={quote_plus(q)}"

def build_department_bubble(row: dict) -> dict:
    def baseline(label: str, value: str) -> dict:
        return {
            "type": "box",
            "layout": "baseline",
            "contents": [
                {"type": "text", "text": label, "size": "sm", "color": "#aaaaaa", "flex": 2},
                {"type": "text", "text": (value.strip() or "—"), "size": "sm", "wrap": True, "flex": 5},
            ],
        }

    dept     = _val(row, "department")        # 標題：系所名稱
    school   = _val(row, "school")            # 次標：學校名稱
    track    = _val(row, "track")             # 分組
    deadline = _val(row, "deadline")          # 截止
    weights  = _val(row, "assessment_weights")# 審查資料（佔比/說明）
    url      = _safe_uri(row)                 # 官方頁面（自動修正成合法 http(s)）

    return {
        "type": "bubble",
        "body": {
            "type": "box",
            "layout": "vertical",
            "contents": [
                {"type": "text", "text": dept, "weight": "bold", "size": "lg", "wrap": True},
                {"type": "text", "text": school, "size": "sm", "color": "#666666", "wrap": True},
                {"type": "separator", "margin": "md"},
                {
                    "type": "box",
                    "layout": "vertical",
                    "margin": "md",
                    "spacing": "sm",
                    "contents": [
                        baseline("分組", track),
                        baseline("截止", deadline),
                        baseline("審查資料", weights),
                    ],
                },
                {
                    "type": "button",
                    "style": "link",
                    "height": "sm",
                    "action": {"type": "uri", "label": "官方頁面", "uri": url},
                },
            ],
        },
        "footer": {
            "type": "box",
            "layout": "vertical",
            "spacing": "sm",
            "contents": [
                {"type": "text", "text": "以官方公告為準", "size": "xs", "color": "#aaaaaa", "wrap": True}
            ],
        },
    }
