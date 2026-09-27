"""main.py -- Application entry point.

Creates the FastAPI app, loads shared resources on startup, and wires
everything together with a single line: app.include_router(router).

All API endpoints live in app/routes.py, not here.

Run:
    uvicorn main:app --reload --port 8000
    Then open http://localhost:8000/docs
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI

from app.api import app_state
from app.config import settings
from app.rag_chain import get_embeddings, get_qdrant_client
from app.routes import router
from app.timing import log_time

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load expensive shared objects once at startup, release them at shutdown.

    Code before `yield` runs once when the server starts.
    Code after `yield` runs once when the server shuts down.
    Loading these objects here means every request can reuse them
    instantly instead of re-creating them each time.
    """
    logger.info("Loading embedding model (%s)...", settings.embedding_provider)
    with log_time("loading embedding model"):
        app_state["embeddings"] = get_embeddings(settings.embedding_provider, settings.openai_api_key)

    logger.info("Connecting to Qdrant...")
    with log_time("connecting to Qdrant"):
        app_state["qdrant_client"] = get_qdrant_client(settings.qdrant_url, settings.qdrant_api_key)

    logger.info("Connecting to OpenAI LLM (%s)...", settings.llm_model)
    app_state["llm"] = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)  # Temp folder for uploads

    logger.info(
        "Ready | embeddings=%s | collection=%s",
        settings.embedding_provider,
        settings.qdrant_collection,
    )
    yield  # Server is now running and handling requests

    app_state.clear()  # Release all resources on shutdown
    logger.info("Server shut down.")


app = FastAPI(
    title="RAG API",
    description="Upload files, ask questions, get answers with exact source references.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow a browser served from a different port/origin to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routes defined in app/routes.py.
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
