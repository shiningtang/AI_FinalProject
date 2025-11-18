# scripts/upsert_embeddings_supabase.py
import os
import time
import argparse
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BATCH = int(os.environ.get("BATCH_SIZE", "32"))
EMB_MODEL = os.environ.get("EMB_MODEL", "text-embedding-3-small")

if not SUPABASE_URL or not SUPABASE_KEY or not OPENAI_API_KEY:
    raise SystemExit("Please set SUPABASE_URL, SUPABASE_KEY, and OPENAI_API_KEY in .env")

# Supabase client
from supabase import create_client
sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# OpenAI embed function (supports new SDK or fallback)
try:
    from openai import OpenAI
    oai = OpenAI(api_key=OPENAI_API_KEY)
    def embed_batch(texts):
        resp = oai.embeddings.create(model=EMB_MODEL, input=texts)
        return [d.embedding for d in resp.data]
except Exception:
    import openai
    openai.api_key = OPENAI_API_KEY
    def embed_batch(texts):
        resp = openai.Embedding.create(model=EMB_MODEL, input=texts)
        return [r["embedding"] for r in resp["data"]]

def build_text_for_embedding(row):
    """決定哪些欄位合併成 embedding 文本；可依需要調整欄位"""
    parts = [
        row.get("school","") or "",
        row.get("college","") or "",
        row.get("department","") or "",
        row.get("track","") or "",
        row.get("other_req","") or "",
        row.get("docs_required","") or "",
        row.get("note","") or ""
    ]
    return " ".join([p for p in parts if p]).strip()

def fetch_rows_without_embedding(limit=500):
    """fetch rows where embedding IS NULL (supabase returns list of dicts)"""
    res = sb.table("programs").select("id, school, college, department, track, other_req, docs_required, note").is_("embedding", None).limit(limit).execute()
    data = res.data
    return data or []

def update_batch_embeddings(batch_id_emb):
    """batch_id_emb: list of tuples (id, emb_list)"""
    # Supabase allows update per row; we loop (could be optimized with RPC)
    for _id, emb in batch_id_emb:
        # Update the embedding column with list -> Postgres pgvector accepts array for vector
        r = sb.table("programs").update({"embedding": emb}).eq("id", _id).execute()
        if r.status_code and r.status_code >= 400:
            print("Update error for", _id, r)
    return

def main():
    total_updated = 0
    while True:
        rows = fetch_rows_without_embedding(limit=500)
        if not rows:
            print("No rows without embedding found. Done.")
            break
        print(f"Fetched {len(rows)} rows to process...")
        # process in batches
        for i in range(0, len(rows), BATCH):
            batch = rows[i:i+BATCH]
            texts = [build_text_for_embedding(r) for r in batch]
            # protect empty texts
            for k,txt in enumerate(texts):
                if not txt:
                    texts[k] = " "  # avoid errors with empty string
            embeddings = embed_batch(texts)
            id_emb = [(r["id"], emb) for r, emb in zip(batch, embeddings)]
            update_batch_embeddings(id_emb)
            total_updated += len(id_emb)
            print(f"Upserted embeddings for batch {i//BATCH + 1}, size {len(id_emb)}")
            time.sleep(0.2)
        # loop again to fetch more rows if any
    print("Total updated:", total_updated)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500, help="fetch limit per iteration")
    args = parser.parse_args()
    main()
