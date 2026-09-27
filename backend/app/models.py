"""
models.py

Pydantic request and response models used by FastAPI.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# ======================================================================
# References
# ======================================================================

class Reference(BaseModel):
    """
    Source information returned with an LLM answer.

    Chunk text is intentionally not exposed to the frontend.
    """

    file_name: str
    page_number: int


# ======================================================================
# Query
# ======================================================================

class QueryRequest(BaseModel):
    """Request body for asking a question."""

    session_id: str

    question: str

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


class QueryResponse(BaseModel):
    """Answer returned by the RAG system."""

    session_id: str

    question: str

    answer: str

    references: List[Reference]

    embedding_provider: str


# ======================================================================
# Upload
# ======================================================================

class FileUploadResult(BaseModel):
    """Status of one uploaded file."""

    file_name: str

    doc_id: str

    file_type: str

    status: str

    chunks_created: int = 0

    error: Optional[str] = None


class UploadResponse(BaseModel):
    """Summary of a multiple-file upload."""

    session_id: str

    total_files: int

    successful: int

    failed: int

    results: List[FileUploadResult]


# ======================================================================
# Documents
# ======================================================================

class DocumentInfo(BaseModel):
    """Information about one indexed document."""

    doc_id: str

    file_name: str

    file_type: str

    chunk_count: int

    upload_timestamp: str


class DocumentListResponse(BaseModel):
    """List of documents belonging to a session."""

    session_id: str

    documents: List[DocumentInfo]

    total: int


# ======================================================================
# Sessions
# ======================================================================

class SessionCreateResponse(BaseModel):
    """Response after creating a new session."""

    session_id: str

    created_at: str


class SessionInfo(BaseModel):
    """Summary information about one session."""

    session_id: str

    created_at: str

    document_count: int

    message_count: int


class SessionListResponse(BaseModel):
    """List of available sessions."""

    sessions: List[SessionInfo]

    total: int


class ChatHistoryMessage(BaseModel):
    """One message from session chat history."""

    role: str

    content: str

    references: List[Reference] = []


class SessionDetailResponse(BaseModel):
    """Complete information about one session."""

    session_id: str

    created_at: str

    documents: List[DocumentInfo]

    messages: List[ChatHistoryMessage]


class SessionDeleteResponse(BaseModel):
    """Response after deleting a session."""

    status: str

    session_id: str

    documents_deleted: int

    messages_deleted: int


# ======================================================================
# Existing document deletion
# ======================================================================

class DeleteResponse(BaseModel):
    """Response after deleting a document."""

    status: str

    doc_id: str

    file_name: str

    chunks_deleted: int