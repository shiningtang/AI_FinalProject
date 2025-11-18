# app/ingest.py
import os
import uuid
import argparse
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from app.vector_store import get_vector_store
import csv
import traceback

load_dotenv()

DATA_CSV = os.environ.get("DATA_CSV", "./data/program_chunks_sample.csv")
BATCH = int(os.environ.get("INGEST_BATCH", "64"))

store = get_vector_store()

def read_rows_as_items(path):
    # read csv with encoding fallback, drop Unnamed cols, fillna("")
    enc = "utf-8"
    df = None
    last_exc = None
    for e in ("utf-8", "cp950", "latin1"):
        try:
            df = pd.read_csv(path, dtype=str, encoding=e, engine="python").fillna("")
            enc = e
            break
        except Exception as ex:
            last_exc = ex
            continue
    if df is None:
        raise RuntimeError(f"Failed to read CSV with tried encodings. last error: {last_exc}")

    # drop Unnamed columns (Excel leftovers)
    df = df.loc[:, ~df.columns.str.match("^Unnamed")]

    print(f"Read OK. detected: csv, encoding: {enc}")
    print("Detected headers:", list(df.columns))
    # simple sample preview
    if len(df) > 0:
        print("Sample rows (first 3):")
        print(df.head(3).to_string(index=False))

    items = []
    for _, row in df.iterrows():
        id_ = str(row.get("id", "") or "").strip()
        if id_ == "":
            id_ = uuid.uuid4().hex
        # build text for embedding from key fields (you can adjust)
        text_parts = []
        for c in ["school","college","department","track","docs_required","assessment_weights","other_req","note"]:
            v = row.get(c, "")
            if v and str(v).strip():
                text_parts.append(str(v).strip())
        text = " | ".join(text_parts)
        if not text:
            # ensure non-empty text for embedding
            text = (str(row.get("department","")) or "")[:200] or " "

        # helper: return None for empty to write NULL in DB
        def maybe_none_str(v):
            if v is None:
                return None
            s = str(v).strip()
            return s if s != "" else None

        meta = {
            "school": maybe_none_str(row.get("school","")),
            "college": maybe_none_str(row.get("college","")),
            "department": maybe_none_str(row.get("department","")),
            "track": maybe_none_str(row.get("track","")),
            "deadline": maybe_none_str(row.get("deadline","")),
            "quota": maybe_none_str(row.get("quota","")),
            "assessment_weights": maybe_none_str(row.get("assessment_weights","")),
            "other_req": maybe_none_str(row.get("other_req","")),
            "docs_required": maybe_none_str(row.get("docs_required","")),
            # keep these as text fields per your note
            "interview_required": maybe_none_str(row.get("interview_required","")),
            "written_exam_required": maybe_none_str(row.get("written_exam_required","")),
            "link_apply": maybe_none_str(row.get("link_apply","")),
            "note": maybe_none_str(row.get("note",""))
        }
        items.append((id_, text, meta))
    return items

def ingest(path=DATA_CSV, batch_size=BATCH, write_bad=True):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV/XLSX file not found: {path}")
    print("Reading CSV:", path)
    items = read_rows_as_items(path)
    print("Total rows to ingest:", len(items))

    # if weaviate or other stores need schema creation, call if available
    try:
        if hasattr(store, "create_schema"):
            store.create_schema()
    except Exception as e:
        print("create_schema error:", e)

    bad_rows = []
    bad_errors = []

    for i in tqdm(range(0, len(items), batch_size), desc="Ingest batches"):
        batch = items[i:i+batch_size]
        try:
            store.upsert(batch, batch_size=batch_size)
        except Exception as e:
            # log and fall back to per-row insert to isolate bad rows
            print(f"Error ingesting batch starting at {i}: {e}\n")
            tb = traceback.format_exc()
            for j, item in enumerate(batch):
                try:
                    store.upsert([item], batch_size=1)
                except Exception as e2:
                    print(f"  Row failed id={item[0]}: {e2}")
                    bad_rows.append(item)
                    bad_errors.append({"id": item[0], "error": str(e2)})
    print("Ingest finished. Processed:", len(items) - len(bad_rows), "Failures (rows):", len(bad_rows))

    if write_bad and bad_rows:
        os.makedirs("bad_rows", exist_ok=True)
        bad_csv = os.path.join("bad_rows", "bad_rows.csv")
        with open(bad_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id","text","meta"])
            for id_, text, meta in bad_rows:
                writer.writerow([id_, text, str(meta)])
        import json
        with open(os.path.join("bad_rows","bad_rows_errors.json"), "w", encoding="utf-8") as f:
            json.dump(bad_errors, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(bad_rows)} bad rows to {bad_csv} and errors to bad_rows/bad_rows_errors.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=False, default=DATA_CSV)
    parser.add_argument("--batch", type=int, default=BATCH)
    args = parser.parse_args()
    ingest(path=args.csv, batch_size=args.batch)
