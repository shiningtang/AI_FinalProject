# app/config.py
from pydantic import BaseModel
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    # LINE
    line_channel_secret: str = os.getenv("LINE_CHANNEL_SECRET", "")
    line_channel_access_token: str = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

    # Query API (你的檢索+LLM 服務)
    query_api_url: str = os.getenv("QUERY_API_URL", "http://127.0.0.1:8000/query")
    client_api_key: str = os.getenv("API_KEY_FOR_CLIENT", "")  # optional header for internal auth

    # OpenAI / Embedding / Model
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    emb_model: str = os.getenv("EMB_MODEL", "text-embedding-3-small")

    # Database / Vector table (Supabase / Postgres)
    database_url: str = os.getenv("DATABASE_URL", "")
    vector_table: str = os.getenv("VECTOR_TABLE", "programs")

    # 資料來源（URL 優先，其次本機路徑）
    data_csv_url_env: str = os.getenv("DATA_CSV_URL", "")
    data_csv_path_env: str = os.getenv("DATA_CSV_PATH", "")        # 新鍵：主資料 CSV
    local_csv_path_env: str = os.getenv("LOCAL_CSV_PATH", "")      # 舊鍵，相容用
    synonyms_csv_path_env: str = os.getenv("SYNONYMS_CSV_PATH", "")# 全域同義詞 CSV

    # 其他運行參數（可放在此統一調整）
    ingest_batch_size: int = int(os.getenv("INGEST_BATCH", "64"))
    emb_retry_max: int = int(os.getenv("EMB_RETRY_MAX", "3"))

    # ---- 封裝好的取值（建議在程式其他地方都用這些屬性） ----
    @property
    def data_csv_url(self) -> str:
        return self.data_csv_url_env or ""

    @property
    def data_csv_path(self) -> str:
        # 優先順序：DATA_CSV_PATH → LOCAL_CSV_PATH → 預設
        return self.data_csv_path_env or self.local_csv_path_env or "data/programs.csv"

    @property
    def local_csv_path(self) -> str:
        # 保留舊名字給舊程式碼；回傳最終決定的本機路徑
        return self.data_csv_path

    @property
    def synonyms_csv_path(self) -> str:
        # 沒設定就用預設 data/synonyms.csv
        return self.synonyms_csv_path_env or "data/synonyms.csv"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
