"""
src/api/main.py

FastAPI application entry point.

Usage:
    uvicorn src.api.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from src.api.routes import router

app = FastAPI(
    title="Skincare AI API",
    description=(
        "Evidence-based skincare regimen generator. "
        "Analyzes skin photos, retrieves scientific evidence, "
        "generates personalized routines, and validates safety."
    ),
    version="0.1.0",
)

# CORS — configurable via CORS_ORIGINS env var
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
