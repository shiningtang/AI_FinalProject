# test_db_conn.py
import os
import psycopg2
from dotenv import load_dotenv
load_dotenv()
dsn = os.environ.get("DATABASE_URL")
print("DATABASE_URL present:", bool(dsn))
try:
    conn = psycopg2.connect(dsn, sslmode='require')
    print("Connected OK")
    conn.close()
except Exception as e:
    print("Connect failed:", e)
