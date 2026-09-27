"""
routes.py

Route declarations only.

Business logic remains inside api.py.
"""

from fastapi import APIRouter

from app.api import (
    create_session,
    delete_document,
    delete_session,
    get_session,
    health,
    list_documents,
    list_sessions,
    query,
    upload_files,
)

from app.models import (
    DeleteResponse,
    DocumentListResponse,
    QueryResponse,
    SessionCreateResponse,
    SessionDeleteResponse,
    SessionDetailResponse,
    SessionListResponse,
    UploadResponse,
)


router = APIRouter()


# ======================================================================
# Session routes
# ======================================================================

router.add_api_route(
    "/sessions",
    create_session,
    methods=["POST"],
    response_model=SessionCreateResponse,
)

router.add_api_route(
    "/sessions",
    list_sessions,
    methods=["GET"],
    response_model=SessionListResponse,
)

router.add_api_route(
    "/sessions/{session_id}",
    get_session,
    methods=["GET"],
    response_model=SessionDetailResponse,
)

router.add_api_route(
    "/sessions/{session_id}",
    delete_session,
    methods=["DELETE"],
    response_model=SessionDeleteResponse,
)


# ======================================================================
# Session document routes
# ======================================================================

router.add_api_route(
    "/sessions/{session_id}/upload",
    upload_files,
    methods=["POST"],
    response_model=UploadResponse,
)

router.add_api_route(
    "/sessions/{session_id}/documents",
    list_documents,
    methods=["GET"],
    response_model=DocumentListResponse,
)

router.add_api_route(
    "/sessions/{session_id}/documents/{doc_id}",
    delete_document,
    methods=["DELETE"],
    response_model=DeleteResponse,
)


# ======================================================================
# Query
# ======================================================================

router.add_api_route(
    "/sessions/{session_id}/query",
    query,
    methods=["POST"],
    response_model=QueryResponse,
)


# ======================================================================
# Health
# ======================================================================

router.add_api_route(
    "/health",
    health,
    methods=["GET"],
)