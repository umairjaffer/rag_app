"""
main.py

FastAPI application entry point.

Responsibilities:

- Initialize PostgreSQL tables
- Load embedding model
- Connect to Qdrant
- Initialize OpenAI LLM
- Create temporary upload directory
- Register API routes
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI

from app import db_models
from app.api import app_state
from app.config import settings
from app.database import Base, engine
from app.rag_chain import (
    get_embeddings,
    get_qdrant_client,
)
from app.routes import router
from app.timing import log_time


# ----------------------------------------------------------------------
# Logging configuration
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ======================================================================
# Application lifespan
# ======================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize shared resources once when FastAPI starts.
    """

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------

    logger.info(
        "Initializing PostgreSQL database..."
    )

    with log_time(
        "initializing PostgreSQL database"
    ):

        Base.metadata.create_all(
            bind=engine
        )

    logger.info(
        "PostgreSQL database ready."
    )

    # ------------------------------------------------------------------
    # Embedding model
    # ------------------------------------------------------------------

    logger.info(
        "Loading embedding model: %s",
        settings.embedding_provider,
    )

    with log_time(
        "loading embedding model"
    ):

        app_state["embeddings"] = get_embeddings(
            settings.embedding_provider,
            settings.openai_api_key,
        )

    # ------------------------------------------------------------------
    # Qdrant
    # ------------------------------------------------------------------

    logger.info(
        "Connecting to Qdrant..."
    )

    with log_time(
        "connecting to Qdrant"
    ):

        app_state["qdrant_client"] = get_qdrant_client(
            settings.qdrant_url,
            settings.qdrant_api_key,
        )

    # ------------------------------------------------------------------
    # OpenAI LLM
    # ------------------------------------------------------------------

    logger.info(
        "Initializing OpenAI LLM: %s",
        settings.llm_model,
    )

    with log_time(
        "initializing OpenAI LLM"
    ):

        app_state["llm"] = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )

    # ------------------------------------------------------------------
    # Temporary upload directory
    # ------------------------------------------------------------------

    Path(
        settings.upload_dir
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Application ready | embedding=%s | "
        "collection=%s | database=PostgreSQL",
        settings.embedding_provider,
        settings.qdrant_collection,
    )

    yield

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    app_state.clear()

    logger.info(
        "Server shut down."
    )


# ======================================================================
# FastAPI application
# ======================================================================

app = FastAPI(
    title="RAG API",
    description=(
        "Session-based RAG API with PostgreSQL, "
        "Qdrant and OpenAI."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ======================================================================
# CORS
# ======================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================================
# Routes
# ======================================================================

app.include_router(
    router
)


# ======================================================================
# Local execution
# ======================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )