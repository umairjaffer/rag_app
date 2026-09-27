"""Pydantic models that define the shape of every API request and response.

FastAPI uses these classes to:
1. Validate incoming data (a wrong type returns a 422 error automatically).
2. Generate the interactive /docs page automatically.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class Reference(BaseModel):
    """One chunk that was retrieved and used to build an answer."""

    file_name: str          # e.g. "annual_report.pdf"
    file_type: str          # e.g. "pdf", "docx", "txt"
    page_number: int        # Which page/row the chunk came from (1-indexed)
    chunk_index: int        # Position of this chunk within the document
    chunk_text: str         # The exact text that was retrieved
    relevance_score: float  # Cosine similarity score between 0.0 and 1.0


class QueryRequest(BaseModel):
    """What the user sends when asking a question."""

    question: str
    top_k: int = Field(default=5, ge=1, le=20)  # How many chunks to retrieve


class QueryResponse(BaseModel):
    """The generated answer plus every source chunk used to build it."""

    question: str
    answer: str
    references: List[Reference]
    embedding_provider: str


class FileUploadResult(BaseModel):
    """Status of a single file within a batch upload."""

    file_name: str
    doc_id: str
    file_type: str
    status: str                    # "indexed" if successful, "error" otherwise
    chunks_created: int = 0        # How many chunks were stored in Qdrant
    error: Optional[str] = None    # Error message if status == "error"


class UploadResponse(BaseModel):
    """Summary of a batch upload -- one FileUploadResult per file."""

    total_files: int
    successful: int
    failed: int
    results: List[FileUploadResult]


class DocumentInfo(BaseModel):
    """One row in the list of indexed documents."""

    doc_id: str              # ID to use when calling DELETE /documents/{doc_id}
    file_name: str
    file_type: str
    chunk_count: int         # Total chunks stored in Qdrant for this file
    upload_timestamp: str    # ISO timestamp of when it was indexed


class DocumentListResponse(BaseModel):
    """Response for GET /documents."""

    documents: List[DocumentInfo]  # One entry per file (not per chunk)
    total: int


class DeleteResponse(BaseModel):
    """Confirmation that a document was removed."""

    status: str            # Always "deleted" on success
    doc_id: str
    file_name: str
    chunks_deleted: int    # How many Qdrant points were removed
