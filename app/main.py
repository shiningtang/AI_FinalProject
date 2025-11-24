# app/main.py
from fastapi import FastAPI
from .query_api import router as query_router
from .linebot import router as linebot_router

app = FastAPI(title="Admissions RAG API")

# mount routers
app.include_router(query_router)
app.include_router(linebot_router)
