"""
api.py

Business logic for all FastAPI endpoints.

The routes themselves are defined in routes.py.
"""

import logging
import os
import time
from pathlib import Path
from typing import List

from fastapi import Depends, File, HTTPException, UploadFile
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.db_models import (
    ChatMessage,
    Document,
    MessageSource,
    ProcessingLog,
    Session as DBSession,
)
from app.file_uploader import is_supported, load_file
from app.models import (
    ChatHistoryMessage,
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    FileUploadResult,
    QueryRequest,
    QueryResponse,
    Reference,
    SessionCreateResponse,
    SessionDeleteResponse,
    SessionDetailResponse,
    SessionInfo,
    SessionListResponse,
    UploadResponse,
)
from app.rag_chain import (
    compute_doc_id,
    delete_indexed_document,
    delete_session_documents,
    index_documents,
    list_indexed_documents,
    query_with_references,
)


logger = logging.getLogger(__name__)


# ======================================================================
# Shared application state
# ======================================================================

app_state: dict = {}


def get_client() -> QdrantClient:
    """Return shared Qdrant client."""

    return app_state["qdrant_client"]


def get_emb():
    """Return shared embedding model."""

    return app_state["embeddings"]


def get_llm() -> ChatOpenAI:
    """Return shared LLM."""

    return app_state["llm"]


# ======================================================================
# Processing log helper
# ======================================================================

def save_processing_log(
    db: Session,
    session_id: str | None,
    step_name: str,
    duration_seconds: float,
    status: str,
    error_message: str | None = None,
):
    """
    Save one processing step to PostgreSQL.

    The same information is also logged to the application logger.
    """

    log = ProcessingLog(
        session_id=session_id,
        step_name=step_name,
        duration_seconds=round(
            duration_seconds,
            4,
        ),
        status=status,
        error_message=error_message,
    )

    db.add(log)
    db.commit()


# ======================================================================
# Session creation
# ======================================================================

def create_session(
    db: Session = Depends(get_db),
) -> SessionCreateResponse:
    """
    Create a completely new RAG session.
    """

    session = DBSession()

    db.add(session)
    db.commit()
    db.refresh(session)

    logger.info(
        "Created session: %s",
        session.id,
    )

    return SessionCreateResponse(
        session_id=session.id,
        created_at=session.created_at.isoformat(),
    )


# ======================================================================
# Session list
# ======================================================================

def list_sessions(
    db: Session = Depends(get_db),
) -> SessionListResponse:
    """
    Return all sessions.
    """

    sessions = (
        db.query(DBSession)
        .order_by(
            DBSession.created_at.desc()
        )
        .all()
    )

    result = []

    for session in sessions:

        document_count = (
            db.query(func.count(Document.id))
            .filter(
                Document.session_id == session.id
            )
            .scalar()
            or 0
        )

        message_count = (
            db.query(func.count(ChatMessage.id))
            .filter(
                ChatMessage.session_id == session.id
            )
            .scalar()
            or 0
        )

        result.append(
            SessionInfo(
                session_id=session.id,
                created_at=session.created_at.isoformat(),
                document_count=document_count,
                message_count=message_count,
            )
        )

    return SessionListResponse(
        sessions=result,
        total=len(result),
    )


# ======================================================================
# Session existence helper
# ======================================================================

def get_session_or_404(
    session_id: str,
    db: Session,
) -> DBSession:
    """
    Return a session or raise 404.
    """

    session = (
        db.query(DBSession)
        .filter(
            DBSession.id == session_id
        )
        .first()
    )

    if session is None:

        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found.",
        )

    return session


# ======================================================================
# Upload files into session
# ======================================================================

async def upload_files(
    session_id: str,
    files: List[UploadFile] = File(
        ...,
        description=(
            "One or more files: PDF, DOCX, MD, CSV, TXT"
        ),
    ),
    client: QdrantClient = Depends(get_client),
    embeddings=Depends(get_emb),
    db: Session = Depends(get_db),
) -> UploadResponse:
    """
    Upload one or multiple files into a specific session.

    A failure in one file does not stop the remaining files.
    """

    # Verify session.
    get_session_or_404(
        session_id,
        db,
    )

    if not files:

        raise HTTPException(
            status_code=400,
            detail="No files provided.",
        )

    request_start = time.perf_counter()

    results: List[FileUploadResult] = []

    docs_to_index: dict = {}

    # ------------------------------------------------------------------
    # Phase 1: Load files
    # ------------------------------------------------------------------

    for upload in files:

        file_name = (
            upload.filename
            or "unknown"
        )

        # --------------------------------------------------------------
        # Validate extension
        # --------------------------------------------------------------

        if not is_supported(file_name):

            results.append(
                FileUploadResult(
                    file_name=file_name,
                    doc_id=compute_doc_id(
                        file_name
                    ),
                    file_type=Path(
                        file_name
                    ).suffix.lstrip("."),
                    status="error",
                    error=(
                        f"Unsupported type "
                        f"'{Path(file_name).suffix}'. "
                        "Accepted PDF, DOCX, MD, CSV, TXT."
                    ),
                )
            )

            continue

        # --------------------------------------------------------------
        # Read uploaded file
        # --------------------------------------------------------------

        load_start = time.perf_counter()

        try:

            content = await upload.read()

            size_mb = (
                len(content)
                / (1024 * 1024)
            )

            if size_mb > settings.max_upload_size_mb:

                elapsed = (
                    time.perf_counter()
                    - load_start
                )

                save_processing_log(
                    db=db,
                    session_id=session_id,
                    step_name=f"loading {file_name}",
                    duration_seconds=elapsed,
                    status="error",
                    error_message=(
                        f"File too large "
                        f"({size_mb:.1f} MB). "
                        f"Maximum allowed: "
                        f"{settings.max_upload_size_mb} MB."
                    ),
                )

                results.append(
                    FileUploadResult(
                        file_name=file_name,
                        doc_id=compute_doc_id(
                            file_name
                        ),
                        file_type=Path(
                            file_name
                        ).suffix.lstrip("."),
                        status="error",
                        error=(
                            f"File too large "
                            f"({size_mb:.1f} MB). "
                            f"Max allowed: "
                            f"{settings.max_upload_size_mb} MB."
                        ),
                    )
                )

                continue

            # ----------------------------------------------------------
            # Save temporary file
            # ----------------------------------------------------------

            tmp_path = os.path.join(
                settings.upload_dir,
                (
                    f"{session_id}_"
                    f"{compute_doc_id(file_name)}_"
                    f"{file_name}"
                ),
            )

            try:

                with open(
                    tmp_path,
                    "wb",
                ) as file:

                    file.write(content)

                docs_to_index[file_name] = load_file(
                    tmp_path,
                    file_name,
                )

                elapsed = (
                    time.perf_counter()
                    - load_start
                )

                save_processing_log(
                    db=db,
                    session_id=session_id,
                    step_name=f"loading {file_name}",
                    duration_seconds=elapsed,
                    status="success",
                )

            finally:

                if os.path.exists(tmp_path):

                    os.unlink(tmp_path)

        except Exception as exc:

            elapsed = (
                time.perf_counter()
                - load_start
            )

            logger.exception(
                "Failed to load '%s'",
                file_name,
            )

            save_processing_log(
                db=db,
                session_id=session_id,
                step_name=f"loading {file_name}",
                duration_seconds=elapsed,
                status="error",
                error_message=str(exc),
            )

            results.append(
                FileUploadResult(
                    file_name=file_name,
                    doc_id=compute_doc_id(
                        file_name
                    ),
                    file_type=Path(
                        file_name
                    ).suffix.lstrip("."),
                    status="error",
                    error=str(exc),
                )
            )

    # ------------------------------------------------------------------
    # Phase 2: Index successfully loaded files
    # ------------------------------------------------------------------

    if docs_to_index:

        index_start = time.perf_counter()

        try:

            index_results = index_documents(
                docs_by_file=docs_to_index,
                client=client,
                embeddings=embeddings,
                collection_name=settings.qdrant_collection,
                session_id=session_id,
            )

            index_elapsed = (
                time.perf_counter()
                - index_start
            )

            save_processing_log(
                db=db,
                session_id=session_id,
                step_name="embedding and Qdrant indexing",
                duration_seconds=index_elapsed,
                status="success",
            )

            # ----------------------------------------------------------
            # Save document metadata in PostgreSQL
            # ----------------------------------------------------------

            for file_name, info in index_results.items():

                doc_id = compute_doc_id(
                    file_name
                )

                # Remove existing PostgreSQL record for
                # the same file inside this session.
                existing_document = (
                    db.query(Document)
                    .filter(
                        Document.session_id == session_id,
                        Document.doc_id == doc_id,
                    )
                    .first()
                )

                if existing_document:

                    db.delete(
                        existing_document
                    )

                    db.commit()

                document = Document(
                    session_id=session_id,
                    doc_id=doc_id,
                    file_name=file_name,
                    file_type=Path(
                        file_name
                    ).suffix.lstrip("."),
                    chunk_count=int(
                        info["chunks_created"]
                    ),
                )

                db.add(document)

                results.append(
                    FileUploadResult(
                        file_name=file_name,
                        doc_id=doc_id,
                        file_type=Path(
                            file_name
                        ).suffix.lstrip("."),
                        status="indexed",
                        chunks_created=int(
                            info["chunks_created"]
                        ),
                    )
                )

            db.commit()

        except ValueError as exc:

            index_elapsed = (
                time.perf_counter()
                - index_start
            )

            save_processing_log(
                db=db,
                session_id=session_id,
                step_name="embedding and Qdrant indexing",
                duration_seconds=index_elapsed,
                status="error",
                error_message=str(exc),
            )

            logger.error(
                "Indexing validation error: %s",
                exc,
            )

            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

        except Exception as exc:

            index_elapsed = (
                time.perf_counter()
                - index_start
            )

            save_processing_log(
                db=db,
                session_id=session_id,
                step_name="embedding and Qdrant indexing",
                duration_seconds=index_elapsed,
                status="error",
                error_message=str(exc),
            )

            logger.exception(
                "Indexing error"
            )

            raise HTTPException(
                status_code=500,
                detail=f"Indexing failed: {exc}",
            )

    elapsed = (
        time.perf_counter()
        - request_start
    )

    logger.info(
        "Upload request for session %s "
        "with %d file(s) took %.2f seconds",
        session_id,
        len(files),
        elapsed,
    )

    return UploadResponse(
        session_id=session_id,
        total_files=len(files),
        successful=sum(
            1
            for result in results
            if result.status == "indexed"
        ),
        failed=sum(
            1
            for result in results
            if result.status == "error"
        ),
        results=results,
    )


# ======================================================================
# Query
# ======================================================================

def query(
    request: QueryRequest,
    client: QdrantClient = Depends(get_client),
    embeddings=Depends(get_emb),
    llm: ChatOpenAI = Depends(get_llm),
    db: Session = Depends(get_db),
) -> QueryResponse:
    """
    Ask a question inside a specific session.

    The question and answer are persisted in PostgreSQL.
    """

    # Verify session.
    get_session_or_404(
        request.session_id,
        db,
    )

    logger.info(
        "Query for session %s: %r",
        request.session_id,
        request.question,
    )

    start_time = time.perf_counter()

    try:

        result = query_with_references(
            question=request.question,
            top_k=request.top_k,
            client=client,
            embeddings=embeddings,
            llm=llm,
            collection_name=settings.qdrant_collection,
            session_id=request.session_id,
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        save_processing_log(
            db=db,
            session_id=request.session_id,
            step_name="query and LLM generation",
            duration_seconds=elapsed,
            status="success",
        )

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        save_processing_log(
            db=db,
            session_id=request.session_id,
            step_name="query and LLM generation",
            duration_seconds=elapsed,
            status="error",
            error_message=str(exc),
        )

        logger.exception(
            "Query error"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    # ------------------------------------------------------------------
    # Save user message
    # ------------------------------------------------------------------

    user_message = ChatMessage(
        session_id=request.session_id,
        role="user",
        content=request.question,
    )

    db.add(user_message)

    db.commit()

    # ------------------------------------------------------------------
    # Save assistant message
    # ------------------------------------------------------------------

    assistant_message = ChatMessage(
        session_id=request.session_id,
        role="assistant",
        content=result["answer"],
    )

    db.add(assistant_message)

    db.commit()

    db.refresh(
        assistant_message
    )

    # ------------------------------------------------------------------
    # Save references
    # ------------------------------------------------------------------

    for reference in result["references"]:

        source = MessageSource(
            message_id=assistant_message.id,
            file_name=reference.file_name,
            page_number=reference.page_number,
        )

        db.add(source)

    db.commit()

    return QueryResponse(
        session_id=request.session_id,
        question=request.question,
        answer=result["answer"],
        references=result["references"],
        embedding_provider=settings.embedding_provider,
    )


# ======================================================================
# Session documents
# ======================================================================

def list_documents(
    session_id: str,
    client: QdrantClient = Depends(get_client),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """
    List documents belonging to the current session.
    """

    get_session_or_404(
        session_id,
        db,
    )

    try:

        documents = list_indexed_documents(
            client=client,
            collection_name=settings.qdrant_collection,
            session_id=session_id,
        )

        return DocumentListResponse(
            session_id=session_id,
            documents=[
                DocumentInfo(**document)
                for document in documents
            ],
            total=len(documents),
        )

    except Exception as exc:

        logger.exception(
            "Document list error"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ======================================================================
# Delete one document
# ======================================================================

def delete_document(
    session_id: str,
    doc_id: str,
    client: QdrantClient = Depends(get_client),
    db: Session = Depends(get_db),
) -> DeleteResponse:
    """
    Delete one document from the current session.
    """

    get_session_or_404(
        session_id,
        db,
    )

    document = (
        db.query(Document)
        .filter(
            Document.session_id == session_id,
            Document.doc_id == doc_id,
        )
        .first()
    )

    file_name = (
        document.file_name
        if document
        else doc_id
    )

    try:

        deleted_count = delete_indexed_document(
            doc_id=doc_id,
            client=client,
            collection_name=settings.qdrant_collection,
            session_id=session_id,
        )

        if deleted_count == 0:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"No document found with "
                    f"doc_id='{doc_id}'"
                ),
            )

        if document:

            db.delete(
                document
            )

            db.commit()

        logger.info(
            "Deleted '%s': %d chunk(s)",
            file_name,
            deleted_count,
        )

        return DeleteResponse(
            status="deleted",
            doc_id=doc_id,
            file_name=file_name,
            chunks_deleted=deleted_count,
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Delete error"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ======================================================================
# Session detail / chat history
# ======================================================================

def get_session(
    session_id: str,
    db: Session = Depends(get_db),
) -> SessionDetailResponse:
    """
    Return documents and complete chat history of a session.
    """

    session = get_session_or_404(
        session_id,
        db,
    )

    documents = (
        db.query(Document)
        .filter(
            Document.session_id == session_id
        )
        .order_by(
            Document.upload_timestamp.desc()
        )
        .all()
    )

    messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id == session_id
        )
        .order_by(
            ChatMessage.created_at.asc()
        )
        .all()
    )

    history = []

    for message in messages:

        references = [
            Reference(
                file_name=source.file_name,
                page_number=source.page_number,
            )
            for source in message.sources
        ]

        history.append(
            ChatHistoryMessage(
                role=message.role,
                content=message.content,
                references=references,
            )
        )

    document_info = [
        DocumentInfo(
            doc_id=document.doc_id,
            file_name=document.file_name,
            file_type=document.file_type,
            chunk_count=document.chunk_count,
            upload_timestamp=(
                document.upload_timestamp.isoformat()
            ),
        )
        for document in documents
    ]

    return SessionDetailResponse(
        session_id=session.id,
        created_at=session.created_at.isoformat(),
        documents=document_info,
        messages=history,
    )


# ======================================================================
# Delete session
# ======================================================================

def delete_session(
    session_id: str,
    client: QdrantClient = Depends(get_client),
    db: Session = Depends(get_db),
) -> SessionDeleteResponse:
    """
    Delete a complete session.

    This removes:

    - Qdrant chunks
    - PostgreSQL documents
    - PostgreSQL chat history
    - PostgreSQL references
    - PostgreSQL processing logs
    """

    session = get_session_or_404(
        session_id,
        db,
    )

    # Count documents/messages before deletion
    document_count = (
        db.query(func.count(Document.id))
        .filter(
            Document.session_id == session_id
        )
        .scalar()
        or 0
    )

    message_count = (
        db.query(func.count(ChatMessage.id))
        .filter(
            ChatMessage.session_id == session_id
        )
        .scalar()
        or 0
    )

    try:

        # --------------------------------------------------------------
        # Delete vectors from Qdrant
        # --------------------------------------------------------------

        qdrant_deleted = delete_session_documents(
            session_id=session_id,
            client=client,
            collection_name=settings.qdrant_collection,
        )

        # --------------------------------------------------------------
        # Delete PostgreSQL session.
        #
        # Relationships use cascade="all, delete-orphan", so related
        # documents/messages/references/logs are removed as well.
        # --------------------------------------------------------------

        db.delete(
            session
        )

        db.commit()

        logger.info(
            "Deleted session %s | Qdrant chunks=%d | "
            "documents=%d | messages=%d",
            session_id,
            qdrant_deleted,
            document_count,
            message_count,
        )

        return SessionDeleteResponse(
            status="deleted",
            session_id=session_id,
            documents_deleted=document_count,
            messages_deleted=message_count,
        )

    except Exception as exc:

        db.rollback()

        logger.exception(
            "Session deletion error"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ======================================================================
# Health
# ======================================================================

def health() -> dict:
    """
    Basic application health endpoint.
    """

    return {
        "status": "ok",
        "embedding_provider": settings.embedding_provider,
        "llm_model": settings.llm_model,
        "collection": settings.qdrant_collection,
        "database": "PostgreSQL",
    }