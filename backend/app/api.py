"""api.py -- The actual logic for every API endpoint.

routes.py imports these functions and registers them on the router.
"""

import logging
import os
import time
from pathlib import Path
from typing import List

from fastapi import Depends, File, HTTPException, UploadFile
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient

from app.config import settings
from app.file_uploader import is_supported, load_file
from app.models import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    FileUploadResult,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.rag_chain import (
    compute_doc_id,
    delete_indexed_document,
    index_documents,
    list_indexed_documents,
    query_with_references,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Shared state & dependencies
#
# app_state is populated once by the lifespan function in main.py at
# startup. The get_* functions below are FastAPI dependencies -- FastAPI
# calls them automatically via Depends() to inject the shared objects
# into each handler.
# --------------------------------------------------------------------------

app_state: dict = {}  # Holds embeddings, qdrant_client, llm -- built once at startup


def get_client() -> QdrantClient:
    """Return the shared Qdrant connection."""
    return app_state["qdrant_client"]


def get_emb():
    """Return the shared embedding model."""
    return app_state["embeddings"]


def get_llm() -> ChatOpenAI:
    """Return the shared OpenAI LLM."""
    return app_state["llm"]


# --------------------------------------------------------------------------
# Upload handler
# --------------------------------------------------------------------------

async def upload_files(
    files: List[UploadFile] = File(..., description="One or more files: PDF, DOCX, MD, CSV, TXT"),
    client: QdrantClient = Depends(get_client),
    embeddings=Depends(get_emb),
) -> UploadResponse:
    """Upload one or many files.

    Each file is loaded, split into chunks, embedded, and stored in
    Qdrant. Returns a per-file status -- one failure does not cancel
    the rest.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    start_time = time.perf_counter()
    results: List[FileUploadResult] = []
    docs_to_index: dict = {}  # {file_name: [Document, ...]}

    # ---- Phase 1: load each file from the uploaded bytes ------------------
    for upload in files:
        file_name = upload.filename or "unknown"

        # Reject unsupported types before doing any disk I/O.
        if not is_supported(file_name):
            results.append(FileUploadResult(
                file_name=file_name,
                doc_id=compute_doc_id(file_name),
                file_type=Path(file_name).suffix.lstrip("."),
                status="error",
                error=f"Unsupported type '{Path(file_name).suffix}'. Accepted PDF, DOCX, MD, CSV, TXT",
            ))
            continue

        content = await upload.read()
        size_mb = len(content) / (1024 * 1024)

        if size_mb > settings.max_upload_size_mb:
            results.append(FileUploadResult(
                file_name=file_name,
                doc_id=compute_doc_id(file_name),
                file_type=Path(file_name).suffix.lstrip("."),
                status="error",
                error=f"File too large ({size_mb:.1f} MB). Max allowed: {settings.max_upload_size_mb} MB",
            ))
            continue

        # LangChain loaders need a real file on disk, so save the bytes
        # to a temp path and delete it again once loading is done.
        tmp_path = os.path.join(settings.upload_dir, f"{compute_doc_id(file_name)}_{file_name}")
        try:
            with open(tmp_path, "wb") as f:
                f.write(content)
            docs_to_index[file_name] = load_file(tmp_path, file_name)

        except Exception as e:
            logger.error("Failed to load '%s': %s", file_name, e)
            results.append(FileUploadResult(
                file_name=file_name,
                doc_id=compute_doc_id(file_name),
                file_type=Path(file_name).suffix.lstrip("."),
                status="error",
                error=str(e),
            ))

        finally:
            # Documents are already in memory, so the temp file is no
            # longer needed either way.
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ---- Phase 2: embed and store all successfully loaded files -----------
    if docs_to_index:
        try:
            index_results = index_documents(
                docs_by_file=docs_to_index,
                client=client,
                embeddings=embeddings,
                collection_name=settings.qdrant_collection,
            )
            for file_name, info in index_results.items():
                results.append(FileUploadResult(
                    file_name=file_name,
                    doc_id=compute_doc_id(file_name),
                    file_type=Path(file_name).suffix.lstrip("."),
                    status="indexed",
                    chunks_created=info["chunks_created"],
                ))

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))  # e.g. embedding dimension mismatch
        except Exception as e:
            logger.error("Indexing error: %s", e)
            raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    elapsed = time.perf_counter() - start_time
    logger.info("Upload request for %d file(s) took %.2f s", len(files), elapsed)

    return UploadResponse(
        total_files=len(files),
        successful=sum(1 for r in results if r.status == "indexed"),
        failed=sum(1 for r in results if r.status == "error"),
        results=results,
    )


# --------------------------------------------------------------------------
# Query handler
# --------------------------------------------------------------------------

def query(
    request: QueryRequest,
    client: QdrantClient = Depends(get_client),
    embeddings=Depends(get_emb),
    llm: ChatOpenAI = Depends(get_llm),
) -> QueryResponse:
    """Ask a question -- returns the answer and every source chunk used to build it."""
    logger.info("Query: %r (top_k=%d)", request.question, request.top_k)
    start_time = time.perf_counter()

    try:
        result = query_with_references(
            question=request.question,
            top_k=request.top_k,
            client=client,
            embeddings=embeddings,
            llm=llm,
            collection_name=settings.qdrant_collection,
        )
    except Exception as e:
        logger.error("Query error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = time.perf_counter() - start_time
    logger.info("Query answered in %.2f s", elapsed)

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        references=result["references"],
        embedding_provider=settings.embedding_provider,
    )


# --------------------------------------------------------------------------
# List documents handler
# --------------------------------------------------------------------------

def list_documents(client: QdrantClient = Depends(get_client)) -> DocumentListResponse:
    """List all files currently indexed in Qdrant (one row per file, not per chunk)."""
    try:
        docs = list_indexed_documents(client, settings.qdrant_collection)
        return DocumentListResponse(
            documents=[DocumentInfo(**d) for d in docs],
            total=len(docs),
        )
    except Exception as e:
        logger.error("List error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------
# Delete document handler
# --------------------------------------------------------------------------

def delete_document(doc_id: str, client: QdrantClient = Depends(get_client)) -> DeleteResponse:
    """Remove a file and all its chunks from Qdrant by doc_id."""
    try:
        # Look up the file name before deleting so we can include it in the response.
        all_docs = list_indexed_documents(client, settings.qdrant_collection)
        doc_info = next((d for d in all_docs if d["doc_id"] == doc_id), None)

        n_deleted = delete_indexed_document(doc_id, client, settings.qdrant_collection)
        if n_deleted == 0:
            raise HTTPException(status_code=404, detail=f"No document found with doc_id='{doc_id}'")

        file_name = doc_info["file_name"] if doc_info else doc_id
        logger.info("Deleted '%s': %d chunk(s) removed", file_name, n_deleted)

        return DeleteResponse(
            status="deleted",
            doc_id=doc_id,
            file_name=file_name,
            chunks_deleted=n_deleted,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Delete error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------
# Health check handler
# --------------------------------------------------------------------------

def health() -> dict:
    """Quick check that the server is running, and show the active configuration."""
    return {
        "status": "ok",
        "embedding_provider": settings.embedding_provider,
        "llm_model": settings.llm_model,
        "collection": settings.qdrant_collection,
    }
