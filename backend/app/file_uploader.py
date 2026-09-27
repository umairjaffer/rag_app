"""Loads uploaded files (PDF, DOCX, Markdown, CSV, TXT) into LangChain
Document objects so they can be split and embedded.
"""

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from app.timing import log_time

logger = logging.getLogger(__name__)

# File extensions we know how to read.
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".csv", ".txt"}


def is_supported(file_name: str) -> bool:
    """Return True if we have a loader for this file's extension."""
    return Path(file_name).suffix.lower() in SUPPORTED_EXTENSIONS


def load_file(file_path: str, file_name: str) -> List[Document]:
    """Load a file from disk into a list of LangChain Documents.

    Every returned Document is stamped with `file_name` and `file_type`
    in its metadata, so this information travels with every chunk all
    the way into Qdrant.

    Args:
        file_path: Where the file currently sits on disk (temp path).
        file_name: The original file name, used for metadata and errors.

    Returns:
        A list of Document objects (one per page, row, or section).

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = Path(file_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{ext}'. Accepted: {allowed}")

    logger.info("Loading '%s' (type: %s)", file_name, ext)

    with log_time(f"loading '{file_name}'"):
        docs = _load_by_extension(file_path, ext)

    for doc in docs:
        doc.metadata["file_name"] = file_name
        doc.metadata["file_type"] = ext.lstrip(".")

    logger.info("-> %d document(s) loaded from '%s'", len(docs), file_name)
    return docs


def _load_by_extension(file_path: str, ext: str) -> List[Document]:
    """Pick the right LangChain loader for `ext` and return raw documents."""
    if ext == ".pdf":
        docs = PyPDFLoader(file_path).load()
        for doc in docs:
            # PyPDFLoader pages are 0-indexed; make them 1-indexed for humans.
            doc.metadata["page_number"] = doc.metadata.get("page", 0) + 1
        return docs

    if ext == ".docx":
        docs = Docx2txtLoader(file_path).load()  # One Document for the whole file
        for doc in docs:
            doc.metadata["page_number"] = 1  # DOCX has no page concept
        return docs

    if ext in (".md", ".markdown"):
        try:
            docs = UnstructuredMarkdownLoader(file_path).load()
        except (ImportError, ModuleNotFoundError):
            # Fall back to plain text if the optional "unstructured"
            # package is not installed.
            docs = TextLoader(file_path, encoding="utf-8").load()
        for doc in docs:
            doc.metadata["page_number"] = 1
        return docs

    if ext == ".csv":
        docs = CSVLoader(file_path, encoding="utf-8").load()
        for i, doc in enumerate(docs):
            doc.metadata["page_number"] = i + 1  # One "page" per row
        return docs

    if ext == ".txt":
        docs = TextLoader(file_path, encoding="utf-8").load()
        for doc in docs:
            doc.metadata["page_number"] = 1
        return docs

    raise ValueError(f"No loader implemented for extension: {ext}")
