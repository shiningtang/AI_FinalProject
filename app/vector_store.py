# app/vector_store.py
import os
import time
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
VECTOR_TABLE = os.environ.get("VECTOR_TABLE", "programs")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")
RETRY_MAX = int(os.environ.get("EMB_RETRY_MAX", "3"))

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in .env")

# OpenAI embed wrapper (with retry)
def get_openai_embed_fn():
    try:
        from openai import OpenAI as OpenAI_NewSDK
        oai = OpenAI_NewSDK(api_key=OPENAI_API_KEY)
        def embed_fn(texts, model=EMB_MODEL):
            last_exc = None
            for attempt in range(1, RETRY_MAX+1):
                try:
                    resp = oai.embeddings.create(model=model, input=texts)
                    return [d.embedding for d in resp.data]
                except Exception as e:
                    last_exc = e
                    msg = str(e).lower()
                    if "quota" in msg or "invalid_api_key" in msg or "insufficient_quota" in msg:
                        raise RuntimeError(f"OpenAI quota/auth error: {e}")
                    time.sleep(0.5 * (2 ** (attempt-1)))
            raise RuntimeError(f"OpenAI embedding failed: {last_exc}")
        return embed_fn
    except Exception:
        try:
            import openai as openai_legacy
            openai_legacy.api_key = OPENAI_API_KEY
            def embed_fn(texts, model=EMB_MODEL):
                last_exc = None
                for attempt in range(1, RETRY_MAX+1):
                    try:
                        resp = openai_legacy.Embedding.create(model=model, input=texts)
                        return [r['embedding'] for r in resp['data']]
                    except Exception as e:
                        last_exc = e
                        msg = str(e).lower()
                        if "quota" in msg or "invalid_api_key" in msg or "insufficient_quota" in msg:
                            raise RuntimeError(f"OpenAI quota/auth error: {e}")
                        time.sleep(0.5 * (2 ** (attempt-1)))
                raise RuntimeError(f"OpenAI embedding failed: {last_exc}")
            return embed_fn
        except Exception:
            raise RuntimeError("OpenAI SDK not available or OPENAI_API_KEY not set")

embed_fn = get_openai_embed_fn()

# Postgres (psycopg2) implementation
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from datetime import datetime, date

def try_parse_date(s):
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    s = s.replace("．", ".").replace("／", "/")
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y/%m", "%Y-%m", "%Y%m%d", "%Y"]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.date()
        except Exception:
            continue
    try:
        parts = [p for p in ''.join(ch if ch.isdigit() or ch in "/-." else " " for ch in s).split() if p]
        if len(parts) == 1:
            p = parts[0].replace(".", "/").replace("-", "/")
            sub = p.split("/")
            if len(sub) >= 1 and len(sub[0]) == 4:
                if len(sub) == 1:
                    return date(int(sub[0]), 1, 1)
                if len(sub) == 2:
                    return date(int(sub[0]), int(sub[1]), 1)
                if len(sub) >= 3:
                    return date(int(sub[0]), int(sub[1]), int(sub[2]))
    except Exception:
        pass
    return None

def safe_int(v):
    try:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return None
        return int(float(v))
    except Exception:
        return None

def maybe_str_field(v):
    if v is None:
        return None
    s = str(v).strip()
    return s if s != "" else None

class SupabaseVectorStore:
    def __init__(self, embedding_fn, table_name=VECTOR_TABLE):
        self.embedding_fn = embedding_fn
        self.table = table_name
        self.dsn = DATABASE_URL

    def upsert(self, items, batch_size=64):
        """
        items: list of (id, text, meta)
        meta fields expected (strings where appropriate). This version
        preserves interview_required and written_exam_required as text fields.
        """
        if not items:
            return
        conn = psycopg2.connect(self.dsn)
        try:
            for i in range(0, len(items), batch_size):
                batch = items[i:i+batch_size]
                texts = [it[1] for it in batch]
                vecs = self.embedding_fn(texts)
                non_vector_rows = []
                rows_with_vec = []
                for (id_, text, meta), vec in zip(batch, vecs):
                    props = dict(meta) if meta else {}
                    deadline_val = try_parse_date(props.get("deadline"))
                    quota_val = safe_int(props.get("quota"))
                    # Keep interview/written as text
                    interview_text = maybe_str_field(props.get("interview_required"))
                    written_text = maybe_str_field(props.get("written_exam_required"))
                    school = maybe_str_field(props.get("school"))
                    college = maybe_str_field(props.get("college"))
                    department = maybe_str_field(props.get("department"))
                    track = maybe_str_field(props.get("track"))
                    assessment_weights = maybe_str_field(props.get("assessment_weights"))
                    other_req = maybe_str_field(props.get("other_req"))
                    docs_required = maybe_str_field(props.get("docs_required"))
                    link_apply = maybe_str_field(props.get("link_apply"))
                    note = maybe_str_field(props.get("note"))

                    non_vector_rows.append((
                        id_,
                        school,
                        college,
                        department,
                        track,
                        deadline_val,
                        quota_val,
                        assessment_weights,
                        other_req,
                        docs_required,
                        interview_text,
                        written_text,
                        link_apply,
                        note
                    ))
                    rows_with_vec.append((id_, vec))
                insert_sql = f"""
                    INSERT INTO {self.table}
                    (id, school, college, department, track, deadline, quota, assessment_weights, other_req, docs_required,
                     interview_required, written_exam_required, link_apply, note)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                      school = EXCLUDED.school,
                      college = EXCLUDED.college,
                      department = EXCLUDED.department,
                      track = EXCLUDED.track,
                      deadline = EXCLUDED.deadline,
                      quota = EXCLUDED.quota,
                      assessment_weights = EXCLUDED.assessment_weights,
                      other_req = EXCLUDED.other_req,
                      docs_required = EXCLUDED.docs_required,
                      interview_required = EXCLUDED.interview_required,
                      written_exam_required = EXCLUDED.written_exam_required,
                      link_apply = EXCLUDED.link_apply,
                      note = EXCLUDED.note;
                """
                with conn.cursor() as cur:
                    execute_values(cur, insert_sql, non_vector_rows, template=None, page_size=batch_size)
                    conn.commit()
                    # update embeddings
                    for id_, vec in rows_with_vec:
                        emb_literal = "'[" + ",".join(map(str, vec)) + "]'::vector"
                        cur.execute(f"UPDATE {self.table} SET embedding = {emb_literal} WHERE id = %s;", (id_,))
                    conn.commit()
        finally:
            conn.close()

    def query(self, q, top_k=4):
        q_emb = self.embedding_fn([q])[0]
        conn = psycopg2.connect(self.dsn)
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                vec_literal = "'[" + ",".join(map(str, q_emb)) + "]'::vector"
                sql = f"""
                SELECT id, school, college, department, track, deadline, docs_required, link_apply, note,
                       interview_required, written_exam_required,
                       embedding <-> {vec_literal} AS distance
                FROM {self.table}
                WHERE embedding IS NOT NULL
                ORDER BY distance
                LIMIT %s;
                """
                cur.execute(sql, (top_k,))
                rows = cur.fetchall()
                results = []
                for r in rows:
                    meta = {
                        "school": r.get("school"),
                        "college": r.get("college"),
                        "department": r.get("department"),
                        "track": r.get("track"),
                        "deadline": r.get("deadline"),
                        "link_apply": r.get("link_apply"),
                        "interview_required": r.get("interview_required"),
                        "written_exam_required": r.get("written_exam_required")
                    }
                    results.append({
                        "id": r.get("id"),
                        "score": float(r.get("distance")) if r.get("distance") is not None else None,
                        "metadata": meta,
                        "text": r.get("note") or ""
                    })
                return results
        finally:
            conn.close()

def get_vector_store():
    return SupabaseVectorStore(embed_fn, table_name=os.environ.get("VECTOR_TABLE", VECTOR_TABLE))
