# app/data.py
from __future__ import annotations
import io
import os
import re
import unicodedata
import requests
import pandas as pd
from typing import Dict, Set, List
from .config import get_settings

settings = get_settings()

# ---------- helpers ----------
def _norm(s: str) -> str:
    """NFKC 全半形統一 + 小寫 + 去空白"""
    return unicodedata.normalize("NFKC", str(s)).lower().strip()

def _read_csv_from_url_or_path(url: str | None, path: str) -> pd.DataFrame:
    if url:
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except Exception:
            pass  # 失敗就走本地
    return pd.read_csv(path)

# ---------- synonyms (global) ----------
_SYN_GROUPS: Dict[str, Set[str]] = {}
_SYN_LOADED = False

def _add_group(items: List[str]) -> None:
    items_n = sorted({_norm(x) for x in items if str(x).strip()})
    if not items_n:
        return
    group: Set[str] = set(items_n)
    for v in items_n:
        if v in _SYN_GROUPS:
            group |= _SYN_GROUPS[v]
    for v in list(group):
        _SYN_GROUPS[v] = set(group)

def _load_synonyms():
    """
    支援兩種格式：
    1) canonical,variants               （variants 用 | 分隔）
    2) type,key,canon,aliases           （aliases 用 ; 或 | 分隔）
    """
    global _SYN_LOADED
    path = getattr(settings, "synonyms_csv_path", os.getenv("SYNONYMS_CSV_PATH", ""))
    if path and os.path.exists(path):
        df = pd.read_csv(path).fillna("")
        cols = set(df.columns.str.lower())
        if {"canonical", "variants"} <= cols:
            for _, r in df.iterrows():
                can = str(r.get("canonical", "")).strip()
                variants = str(r.get("variants", "")).strip()
                alts = [a.strip() for a in variants.replace("｜","|").split("|") if a.strip()]
                bucket = [can] + alts
                _add_group(bucket)
        elif {"type", "key", "canon", "aliases"} <= cols:
            for _, r in df.iterrows():
                key   = str(r.get("key", "")).strip()
                canon = str(r.get("canon", "")).strip()
                aliases = str(r.get("aliases", "")).strip()
                alts = [a.strip() for a in re.split(r"[;|、/]+", aliases) if a.strip()]
                bucket = ([key] if key else []) + ([canon] if canon else []) + alts
                _add_group(bucket)
    _SYN_LOADED = True

def _ensure_synonyms_loaded():
    if not _SYN_LOADED:
        _load_synonyms()

def _synonyms_for(token: str) -> Set[str]:
    """回傳 token 的所有同義詞（含 token 本身），皆已正規化"""
    _ensure_synonyms_loaded()
    t = _norm(token)
    return _SYN_GROUPS.get(t, {t})

# ---------- data loading ----------
_DF: pd.DataFrame | None = None

def _derive_fields(df: pd.DataFrame) -> pd.DataFrame:
    """將你的欄位補齊/衍生為其他模組會用到的欄位。"""
    if "link_apply" in df.columns and "contact_url" not in df.columns:
        df["contact_url"] = df["link_apply"]
    if "note" in df.columns and "notes" not in df.columns:
        df["notes"] = df["note"]

    other = df["other_req"] if "other_req" in df.columns else ""
    docs  = df["docs_required"] if "docs_required" in df.columns else ""
    req_blob = (other.astype(str).fillna("") + " " + docs.astype(str).fillna("")).astype(str)

    if "english_required" not in df.columns:
        df["english_required"] = req_blob

    if "portfolio_required" not in df.columns:
        df["portfolio_required"] = req_blob.str.contains("作品集", case=False, na=False)\
                                    .map(lambda x: "true" if x else "false")

    if "gpa_min" not in df.columns:
        patt = re.compile(r"gpa[^0-9]*([0-4](?:\.[0-9])?)(?:\s*/\s*([0-4](?:\.[0-9])?))?", re.IGNORECASE)
        def _pick_gpa(s: str) -> str:
            m = patt.search(str(s))
            if not m:
                return ""
            a, b = m.group(1), m.group(2)
            return f"{a}/{b}" if b else a
        df["gpa_min"] = req_blob.map(_pick_gpa)

    if "aliases" not in df.columns:
        df["aliases"] = ""

    return df

def load_dataframe() -> pd.DataFrame:
    """先嘗試 URL，失敗再用本地。"""
    global _DF
    data_url = getattr(settings, "data_csv_url", "") or None
    data_path = (
        getattr(settings, "data_csv_path", "") or
        getattr(settings, "local_csv_path", "") or
        "data/programs.csv"
    )
    df = _read_csv_from_url_or_path(data_url, data_path)

    for col in [
        "school","college","department","track","deadline","quota",
        "other_req","docs_required","link_apply","note",
        "program","degree","english_required","gpa_min","portfolio_required",
        "contact_url","aliases","notes",
        "interview_required","written_exam_required","assessment_weights",
    ]:
        if col not in df.columns:
            df[col] = ""

    df = _derive_fields(df)

    for col in [
        "school","college","department","track","deadline","english_required","gpa_min",
        "portfolio_required","quota","contact_url","aliases","notes",
        "interview_required","written_exam_required","assessment_weights",
    ]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["__search_blob"] = (
        df["school"] + " " + df["college"] + " " + df["department"] + " " +
        df["track"] + " " + df["aliases"]
    ).map(_norm)

    _DF = df
    _ensure_synonyms_loaded()
    return df

# ---------- query parsing & filtering ----------
INTENT_WORDS = {
    "截止","截止日","deadline",
    "名額","人數","quota",
    "比重","權重","審查資料",
    "其他要求","特殊要求",
    "書審","書面資料","文件",
    "面試","口試","面談",
    "筆試","筆測","筆考",
}

def parse_query(q: str) -> dict:
    """判斷意圖 + 取 tokens + 取月份（若有）"""
    qn = _norm(q)

    intent = None
    if any(k in qn for k in ["截止","deadline","截止日"]):
        intent = "deadline"
    elif any(k in qn for k in ["名額","人數","quota"]):
        intent = "quota"
    elif any(k in qn for k in ["比重","權重","審查資料"]):
        intent = "assessment_weights"
    elif any(k in qn for k in ["其他要求","特殊要求"]):
        intent = "other_req"
    elif any(k in qn for k in ["書審","書面資料","文件"]):
        intent = "docs_required"
    elif any(k in qn for k in ["面試","口試","面談"]):
        intent = "interview_required"
    elif any(k in qn for k in ["筆試","筆測","考試"]):
        intent = "written_exam_required"

    month = None
    m = re.search(r"(1[0-2]|0?[1-9])月", qn)
    if m:
        month = int(m.group(1))

    raw_tokens = [t for t in re.split(r"\s+", qn) if t]
    tokens: List[str] = []
    for t in raw_tokens:
        if t in INTENT_WORDS or re.fullmatch(r"(1[0-2]|0?[1-9])月", t):
            continue
        if t not in tokens:
            tokens.append(t)

    return {"intent": intent, "tokens": tokens, "month": month}

def filter_rows(
    df: pd.DataFrame,
    tokens: List[str],
    month: int | None,
    limit: int | None = None
) -> pd.DataFrame:
    """
    在 school/college/department/track/aliases（__search_blob）裡做逐詞 AND，
    每個 token 用其同義詞集合做 OR。
    limit=None 表示不截斷；否則回傳前 limit 筆。
    """
    if tokens:
        mask = pd.Series(True, index=df.index)
        for t in tokens:
            alts = _synonyms_for(t)
            patt = "|".join(re.escape(a) for a in sorted(alts, key=len, reverse=True))
            mask &= df["__search_blob"].str.contains(patt, regex=True, na=False)
        df = df[mask]

    if month:
        m1 = df["deadline"].astype(str).str.contains(fr"-{month:02d}-", na=False)
        m2 = df["deadline"].astype(str).str.contains(fr"/{month}/",   na=False)
        df = df[m1 | m2]

    return df if limit is None else df.head(limit)
