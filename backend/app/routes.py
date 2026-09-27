"""routes.py -- Route declarations only.

Maps every URL path and HTTP method to its handler function in api.py.
No business logic here, just the routing table.
"""

from fastapi import APIRouter

from app.api import (
    delete_document,
    health,
    list_documents,
    query,
    upload_files,
)
from app.models import (
    DeleteResponse,
    DocumentListResponse,
    QueryResponse,
    UploadResponse,
)

router = APIRouter()  # Registered in main.py with app.include_router(router)

# path                          method   handler          response model
router.add_api_route("/upload", upload_files, methods=["POST"], response_model=UploadResponse)
router.add_api_route("/query", query, methods=["POST"], response_model=QueryResponse)
router.add_api_route("/documents", list_documents, methods=["GET"], response_model=DocumentListResponse)
router.add_api_route("/documents/{doc_id}", delete_document, methods=["DELETE"], response_model=DeleteResponse)
router.add_api_route("/health", health, methods=["GET"])
