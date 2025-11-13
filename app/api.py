from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas import IngestResponse, QueryRequest, QueryResponse
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter()
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_200_OK)
def ingest(rag_service: Annotated[RAGService, Depends(get_rag_service)]) -> IngestResponse:
    try:
        stats = rag_service.ingest()
        return IngestResponse(documents_indexed=stats["documents_indexed"])
    except FileNotFoundError as exc:
        logger.error("Ingestion failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Unexpected ingestion error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ingestion failed.") from exc


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def query(
    payload: QueryRequest,
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
) -> QueryResponse:
    try:
        result = rag_service.query(payload.question, similarity_top_k=payload.top_k or 3)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
        )
    except ValueError as exc:
        logger.error("Invalid query payload: %s", exc)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Unexpected query error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Query failed.") from exc

