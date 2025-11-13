from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    status: str = Field(default="ok")
    documents_indexed: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user question to answer.")
    top_k: Optional[int] = Field(default=3, ge=1, le=10, description="Number of similar chunks to retrieve.")


class QuerySource(BaseModel):
    id: str
    score: float
    file_name: Optional[str] = None
    page_number: Optional[str] = None
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[QuerySource]

