"""All RAG logic: embeddings, indexing, querying, and document management.

Pipeline overview
------------------
Upload (indexing):
    1. file_uploader.py                 load pages/rows into Document objects
    2. RecursiveCharacterTextSplitter   split into ~500 character chunks
    3. embeddings.embed_documents()     turn every chunk into a vector (batched)
    4. qdrant_client.upsert()           store vector + metadata in Qdrant

Query:
    1. embeddings.embed_query()         turn the question into a vector
    2. qdrant_client.query_points()     find the top-K most similar chunks
    3. build a context string           combine chunks into one text block
    4. prompt | llm | StrOutputParser   generate the answer
"""

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.models import Reference
from app.timing import log_time

logger = logging.getLogger(__name__)

CHUNK_SIZE = 500     # Max characters per chunk
CHUNK_OVERLAP = 50   # Characters shared between consecutive chunks

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the question using ONLY the "
    "context below.\n"
    "Do NOT mention file names, page numbers, or sources in your answer "
    "- those are shown separately.\n"
    "If the answer is not in the context, say "
    '"I could not find that in the uploaded documents."\n\n'
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

def get_embeddings(provider: str, openai_api_key: str | None = None):
    """Return the embedding model for the given provider.

    "openai"      -> text-embedding-3-small (1536 dims, paid API call)
    "huggingface" -> all-MiniLM-L6-v2 (384 dims, free, runs locally)

    Args:
        provider: "openai" or "huggingface".
        openai_api_key: Required when provider is "openai". Passed in
            explicitly (instead of relying on an OPENAI_API_KEY
            environment variable) because pydantic-settings reads
            .env into `settings` without exporting it to the OS
            environment.
    """
    if provider == "openai":
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# --------------------------------------------------------------------------
# Qdrant helpers
# --------------------------------------------------------------------------

def get_qdrant_client(url: str, api_key: str) -> QdrantClient:
    """Create a Qdrant client connected to the given cluster."""
    return QdrantClient(url=url, api_key=api_key)


def compute_doc_id(file_name: str) -> str:
    """Turn a file name into a short, stable ID (same name -> same ID).

    Every chunk of a file stores this ID, so all of them can be found
    and deleted together with a single filter.
    """
    return hashlib.sha256(file_name.encode()).hexdigest()[:16]


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Return True if the named collection already exists in Qdrant."""
    names = [c.name for c in client.get_collections().collections]
    return collection_name in names


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Create the collection if it does not exist yet, and validate its size.

    Also makes sure a keyword payload index exists on `doc_id`, which is
    required for filtering during delete.

    Raises:
        ValueError: If an existing collection's vector size does not
            match the current embedding model's output size.
    """
    if not collection_exists(client, collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        client.create_payload_index(
            collection_name=collection_name,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Created collection '%s' (%d dims)", collection_name, vector_size)
        return

    # Collection already exists -- make sure the doc_id index is present
    # too (older collections created before this fix may be missing it).
    # Qdrant silently ignores this call if the index already exists.
    client.create_payload_index(
        collection_name=collection_name,
        field_name="doc_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    info = client.get_collection(collection_name)
    vectors_config = info.config.params.vectors
    # Newer qdrant-client returns {"": VectorParams}; older returns VectorParams directly.
    existing_size = (
        list(vectors_config.values())[0].size
        if isinstance(vectors_config, dict)
        else vectors_config.size
    )

    if existing_size != vector_size:
        raise ValueError(
            f"Collection '{collection_name}' was built with {existing_size}-dim "
            f"vectors, but the current embedding model produces {vector_size}-dim "
            f"vectors. Change EMBEDDING_PROVIDER in .env to match, or delete the "
            f"collection in Qdrant Cloud and restart the server."
        )


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------

def index_documents(
    docs_by_file: Dict[str, List[Document]],
    client: QdrantClient,
    embeddings,
    collection_name: str,
) -> Dict[str, Dict[str, object]]:
    """Split, embed, and store documents in Qdrant.

    Handles re-uploads correctly: uploading the same file again replaces
    its old chunks instead of duplicating them.

    Args:
        docs_by_file: {file_name: [Document, ...]}, from file_uploader.load_file().
        client: A connected Qdrant client.
        embeddings: An embedding model, from get_embeddings().
        collection_name: The Qdrant collection to write into.

    Returns:
        {file_name: {"doc_id": str, "chunks_created": int}}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    upload_ts = datetime.now(timezone.utc).isoformat()

    results: Dict[str, Dict[str, object]] = {}
    texts: List[str] = []
    metadata: List[dict] = []

    # Step 1: split every file into chunks, and remember which file and
    # position each chunk came from.
    with log_time("splitting documents into chunks"):
        for file_name, docs in docs_by_file.items():
            doc_id = compute_doc_id(file_name)
            chunks = splitter.split_documents(docs)
            logger.info("'%s': %d doc(s) -> %d raw chunks", file_name, len(docs), len(chunks))

            chunk_count = 0
            for idx, chunk in enumerate(chunks):
                text = chunk.page_content.strip()
                if not text:
                    continue  # Skip blank pages or empty rows

                texts.append(text)
                metadata.append({
                    "file_name": file_name,
                    "file_type": chunk.metadata.get("file_type", "unknown"),
                    "page_number": chunk.metadata.get("page_number", 1),
                    "chunk_index": idx,
                    "doc_id": doc_id,
                    "chunk_text": text,
                    "upload_timestamp": upload_ts,
                })
                chunk_count += 1

            results[file_name] = {"doc_id": doc_id, "chunks_created": chunk_count}

    if not texts:
        logger.warning("No non-empty chunks found to index")
        return results

    # Step 2: embed every chunk in one batch call instead of one at a
    # time -- much faster and more scalable for large uploads.
    with log_time(f"embedding {len(texts)} chunk(s)"):
        vectors = embeddings.embed_documents(texts)

    ensure_collection(client, collection_name, len(vectors[0]))

    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vector, payload=meta)
        for vector, meta in zip(vectors, metadata)
    ]

    # Step 3: remove any older chunks belonging to files being re-uploaded.
    with log_time("removing old chunks for re-uploaded files"):
        for file_name in docs_by_file:
            old_count = delete_indexed_document(compute_doc_id(file_name), client, collection_name)
            if old_count > 0:
                logger.info("Replaced %d old chunk(s) for '%s'", old_count, file_name)

    # Step 4: store every new chunk in a single batch call.
    with log_time(f"storing {len(points)} chunk(s) in Qdrant"):
        client.upsert(collection_name=collection_name, points=points)

    logger.info("Stored %d chunk(s) in '%s'", len(points), collection_name)
    return results


# --------------------------------------------------------------------------
# Querying
# --------------------------------------------------------------------------

def query_with_references(
    question: str,
    top_k: int,
    client: QdrantClient,
    embeddings,
    llm: ChatOpenAI,
    collection_name: str,
) -> dict:
    """Answer a question using the indexed documents.

    Returns:
        {"answer": str, "references": List[Reference]}
    """
    if not collection_exists(client, collection_name):
        return {
            "answer": "No documents have been uploaded yet. Please upload files first.",
            "references": [],
        }

    with log_time("embedding the question"):
        query_vector = embeddings.embed_query(question)

    with log_time("searching Qdrant"):
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

    if not results.points:
        return {
            "answer": "No documents have been uploaded yet. Please upload files first.",
            "references": [],
        }

    references: List[Reference] = []
    context_parts: List[str] = []

    for hit in results.points:
        payload = hit.payload
        references.append(Reference(
            file_name=payload["file_name"],
            file_type=payload.get("file_type", "unknown"),
            page_number=payload.get("page_number", 1),
            chunk_index=payload.get("chunk_index", 0),
            chunk_text=payload["chunk_text"],
            relevance_score=round(hit.score, 4),
        ))
        context_parts.append(
            f"[{payload['file_name']} - page {payload.get('page_number', '?')}]\n"
            f"{payload['chunk_text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    with log_time("generating the answer"):
        chain = ANSWER_PROMPT | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

    return {"answer": answer, "references": references}


# --------------------------------------------------------------------------
# Document management
# --------------------------------------------------------------------------

def list_indexed_documents(client: QdrantClient, collection_name: str) -> List[dict]:
    """Return one summary entry per file (not per chunk).

    Returns an empty list if no files have been uploaded yet.
    """
    if not collection_exists(client, collection_name):
        return []

    seen: Dict[str, dict] = {}   # {doc_id: aggregated info dict}
    offset = None                # Qdrant scroll cursor; None starts from the beginning

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            payload = record.payload
            doc_id = payload.get("doc_id", "unknown")

            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "file_name": payload.get("file_name", "unknown"),
                    "file_type": payload.get("file_type", "unknown"),
                    "upload_timestamp": payload.get("upload_timestamp", ""),
                    "chunk_count": 0,
                }
            seen[doc_id]["chunk_count"] += 1

        if next_offset is None:
            break  # scroll() returns None when there are no more pages
        offset = next_offset

    return list(seen.values())


def delete_indexed_document(doc_id: str, client: QdrantClient, collection_name: str) -> int:
    """Delete every chunk in Qdrant that belongs to this doc_id.

    Returns the number of points deleted (0 if doc_id was not found).
    """
    if not collection_exists(client, collection_name):
        return 0

    doc_filter = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
    count = client.count(collection_name=collection_name, count_filter=doc_filter).count

    if count > 0:
        client.delete(collection_name=collection_name, points_selector=doc_filter)

    return count
