from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api import router as api_router
from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(
    title="Legal RAG Prototype",
    version="0.1.0",
    description="Prototype backend that augments Qwen/Qwen3-8B with POCSO and BNSS Acts.",
)


@app.on_event("startup")
async def _ensure_directories() -> None:
    settings = get_settings()
    settings.resolved_data_dir
    settings.resolved_persist_dir


app.include_router(api_router, prefix="/api")

