# app/query_api.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict

from .services import store, send_llm_prompt, build_prompt, _json_extract, _safe_serialize_value, extract_used_indexes

router = APIRouter()

class QueryIn(BaseModel):
    q: str
    top_k: int = 4

@router.post("/query")
async def query_endpoint(q_in: QueryIn):
    if store is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    if not q_in.q:
        raise HTTPException(status_code=400, detail="Empty query")

    q = q_in.q
    top_k = max(1, min(20, q_in.top_k))

    try:
        retrieved = store.query(q, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {e}")

    prompt = build_prompt(q, retrieved)
    try:
        answer_raw = send_llm_prompt(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    parsed = _json_extract(answer_raw)

    used_idxs = extract_used_indexes(parsed, default_k=top_k, retrieved_len=len(retrieved))

    # build matched_rows from used indices (1-based -> convert to 0-based)
    matched_rows = []
    for idx in used_idxs:
        r = retrieved[idx - 1]
        meta = r.get("metadata") or {}
        safe_meta = {
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
            "note": _safe_serialize_value(meta.get("note", "")),
        }
        matched_rows.append({"row_index": idx, "metadata": safe_meta, "text": r.get("text", "") or ""})

    # answer_struct: combine LLM's short/detail (if any) + matched_rows
    answer_struct = {
        "short_answer": (parsed.get("short_answer") if isinstance(parsed, dict) else "") if parsed else "",
        "detail": (parsed.get("detail") if isinstance(parsed, dict) else "") if parsed else "",
        "matched_rows": matched_rows,
        "llm_parsed": parsed
    }

    return {
        "question": q,
        "answer_raw": answer_raw,
        "answer_struct": answer_struct,
        "sources": matched_rows
    }

# Make call_internal_query importable by linebot module
def call_internal_query(user_text: str, top_k: int = 3) -> Dict:
    in_model = QueryIn(q=user_text, top_k=top_k)
    return (  # reuse same logic by calling the endpoint function synchronously
        # Note: for simplicity, call query_endpoint directly (sync).
        # In larger systems you may refactor common logic into a helper function.
        __import__("asyncio").get_event_loop().run_until_complete(query_endpoint(in_model))
    )
