from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Upload & Document schemas ─────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Returned immediately after POST /api/documents/upload (HTTP 202)."""

    document_id: str = Field(
        ...,
        description="UUID assigned to this upload. Use with /status/{document_id} to poll.",
        examples=["3f2a1b9c-e4d5-4f6a-b7c8-9d0e1f2a3b4c"],
    )
    filename: str = Field(
        ...,
        description="Original filename as uploaded by the user.",
        examples=["annual_report_2024.pdf"],
    )
    chunks_count: int = Field(
        default=0,
        ge=0,
        description="Number of text chunks stored. 0 while status is 'queued' or 'processing'.",
    )
    status: str = Field(
        ...,
        description="One of: queued | processing | completed | duplicate | failed",
        examples=["queued"],
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"queued", "processing", "completed", "duplicate", "failed"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed}, got '{v}'")
        return v


class DocumentInfo(BaseModel):
    """One entry in the GET /api/documents list response."""

    filename: str = Field(
        ...,
        description="Original filename used as the 'source' key in ChromaDB metadata.",
        examples=["legal_contract_v3.docx"],
    )
    document_id: str = Field(
        default="unknown",
        description="UUID of the ingestion run that produced these chunks.",
    )
    chunks_count: int = Field(
        ...,
        ge=0,
        description="Total number of chunks stored in ChromaDB for this document.",
    )
    uploaded_at: Optional[datetime] = Field(
        default=None,
        description=(
            "ISO-8601 timestamp of when the document was ingested. "
            "None when the value is not available in ChromaDB metadata."
        ),
        examples=["2026-03-03T14:22:10Z"],
    )


# ── Query schemas ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Body for POST /api/query."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural-language question to ask about the uploaded documents.",
        examples=["What was the total revenue for FY2023?"],
    )
    document_filter: Optional[str] = Field(
        default=None,
        description=(
            "Optional filename to scope retrieval to a single document. "
            "When None, all ingested documents are searched. "
            "Must exactly match the 'filename' field returned by GET /api/documents."
        ),
        examples=["annual_report_2024.pdf"],
    )

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank or whitespace only.")
        return v.strip()


class SourceChunk(BaseModel):
    """A single retrieved document chunk returned alongside the answer."""

    content: str = Field(
        ...,
        description="Raw text of the retrieved chunk as stored in ChromaDB.",
        examples=["Total revenue for FY2023 was $12.4 billion, up 8% year-over-year."],
    )
    source: str = Field(
        ...,
        description="Filename of the document this chunk was extracted from.",
        examples=["annual_report_2024.pdf"],
    )
    page: int = Field(
        default=0,
        ge=0,
        description="1-based page number within the source document. 0 if unknown.",
        examples=[4],
    )
    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine similarity score in [0, 1]. Higher is more relevant. "
            "None when the retriever does not return scores."
        ),
        examples=[0.87],
    )


class QueryResponse(BaseModel):
    """Response for POST /api/query."""

    answer: str = Field(
        ...,
        description=(
            "LLM-generated answer grounded in the retrieved chunks. "
            "Contains 'I don't have enough information' when no relevant chunks were found."
        ),
        examples=["Total revenue for FY2023 was $12.4 billion (annual_report_2024.pdf, p.4)."],
    )
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description=(
            "Up to 5 source chunks used to generate the answer, "
            "ordered by descending relevance score."
        ),
    )

    @field_validator("sources")
    @classmethod
    def sort_sources_by_score(cls, v: list[SourceChunk]) -> list[SourceChunk]:
        """Always return sources highest-score first; push None scores to the end."""
        return sorted(v, key=lambda s: (s.score is None, -(s.score or 0.0)))