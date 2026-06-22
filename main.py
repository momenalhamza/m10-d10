"""Drill 10 — Typed Retrieval Endpoint.

Build a FastAPI service with a single `/retrieve` POST endpoint that
returns a top-k token-overlap retrieval against an in-memory fixture,
plus a `/healthz` liveness probe.

The drill teaches the typed-boundary contract: Pydantic request and
response models, OpenAPI documentation at `/docs`, and `Field(...)`
constraints that produce 422 at the boundary rather than 500 inside
the function body.
"""
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from retrieval import retrieve_top_k  # pre-implemented; do not modify retrieval.py

app = FastAPI(title="Drill 10 — Typed Retrieval Endpoint")


# --- Pydantic shapes -------------------------------------------------

class RetrieveRequest(BaseModel):
    """Request body for POST /retrieve.

    Inputs:
        query — non-empty search string (1..500 chars).
        k — number of chunks to return (1..10, default 3).
    """
    query: str = Field(..., min_length=1, max_length=500)
    k: int = Field(3, ge=1, le=10)


class Chunk(BaseModel):
    """One retrieved chunk."""
    chunk_id: int
    text: str
    score: float


class RetrieveResponse(BaseModel):
    """Response body for POST /retrieve."""
    retrieved: List[Chunk]


class HealthResponse(BaseModel):
    """Response body for GET /healthz."""
    status: str


# --- Path operations -------------------------------------------------

@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """Retrieve top-k chunks for a query.

    Returns RetrieveResponse with the top-k token-overlap matches. If no
    chunk overlaps, returns 200 with `retrieved=[]` (not an error).
    """
    raw = retrieve_top_k(req.query, req.k)
    return RetrieveResponse(retrieved=[Chunk(**r) for r in raw])


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Liveness probe. Returns 200 with {"status": "ok"}."""
    return HealthResponse(status="ok")
