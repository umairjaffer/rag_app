"""
rag_chain.py

Core RAG functionality:

Upload:
    1. Load documents
    2. Split documents
    3. Generate embeddings
    4. Store vectors and metadata in Qdrant

Query:
    1. Embed question
    2. Retrieve chunks belonging to the current session
    3. Build context
    4. Generate answer using OpenAI

Document management:
    - List documents
    - Delete documents
    - Delete complete session data from Qdrant
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


# ======================================================================
# Chunk configuration
# ======================================================================

CHUNK_SIZE = 500

CHUNK_OVERLAP = 50


# ======================================================================
# Prompt
# ======================================================================

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the question using ONLY "
    "the context below.\n"
    "Do not make up information.\n"
    "Do not mention file names or page numbers in the answer because "
    "they are displayed separately as sources.\n"
    "If the answer is not available in the context, say: "
    '"I could not find that in the uploaded documents."\n\n'
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


# ======================================================================
# Embeddings
# ======================================================================

def get_embeddings(
    provider: str,
    openai_api_key: str | None = None,
):
    """
    Create the configured embedding model.
    """

    if provider == "openai":

        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_api_key,
        )

    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )


# ======================================================================
# Qdrant
# ======================================================================

def get_qdrant_client(
    url: str,
    api_key: str,
) -> QdrantClient:

    return QdrantClient(
        url=url,
        api_key=api_key,
    )


def compute_doc_id(file_name: str) -> str:
    """
    Generate a stable document ID from the file name.
    """

    return hashlib.sha256(
        file_name.encode()
    ).hexdigest()[:16]


def collection_exists(
    client: QdrantClient,
    collection_name: str,
) -> bool:

    names = [
        collection.name
        for collection in client.get_collections().collections
    ]

    return collection_name in names


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """
    Create Qdrant collection if necessary.

    Also creates payload indexes used for session/document filtering.
    """

    if not collection_exists(
        client,
        collection_name,
    ):

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        client.create_payload_index(
            collection_name=collection_name,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

        client.create_payload_index(
            collection_name=collection_name,
            field_name="session_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )

        logger.info(
            "Created Qdrant collection '%s' with %d dimensions",
            collection_name,
            vector_size,
        )

        return

    # Make sure required payload indexes exist.
    client.create_payload_index(
        collection_name=collection_name,
        field_name="doc_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="session_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )

    info = client.get_collection(
        collection_name
    )

    vectors_config = info.config.params.vectors

    existing_size = (
        list(vectors_config.values())[0].size
        if isinstance(vectors_config, dict)
        else vectors_config.size
    )

    if existing_size != vector_size:

        raise ValueError(
            f"Collection '{collection_name}' was built with "
            f"{existing_size}-dim vectors, but the current "
            f"embedding model produces {vector_size}-dim vectors."
        )


# ======================================================================
# Indexing
# ======================================================================

def index_documents(
    docs_by_file: Dict[str, List[Document]],
    client: QdrantClient,
    embeddings,
    collection_name: str,
    session_id: str,
) -> Dict[str, Dict[str, object]]:
    """
    Split, embed and store documents.

    Every chunk receives session_id in its Qdrant payload.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    upload_ts = datetime.now(
        timezone.utc
    ).isoformat()

    results: Dict[str, Dict[str, object]] = {}

    texts: List[str] = []

    metadata: List[dict] = []

    # ------------------------------------------------------------------
    # Step 1: Split documents
    # ------------------------------------------------------------------

    with log_time(
        "splitting documents into chunks"
    ):

        for file_name, docs in docs_by_file.items():

            doc_id = compute_doc_id(
                file_name
            )

            chunks = splitter.split_documents(
                docs
            )

            logger.info(
                "'%s': %d document(s) -> %d chunks",
                file_name,
                len(docs),
                len(chunks),
            )

            chunk_count = 0

            for idx, chunk in enumerate(chunks):

                text = chunk.page_content.strip()

                if not text:
                    continue

                texts.append(text)

                metadata.append(
                    {
                        # Session information is now stored
                        # with every Qdrant point.
                        "session_id": session_id,

                        "file_name": file_name,

                        "file_type": chunk.metadata.get(
                            "file_type",
                            "unknown",
                        ),

                        "page_number": chunk.metadata.get(
                            "page_number",
                            1,
                        ),

                        "chunk_index": idx,

                        "doc_id": doc_id,

                        "chunk_text": text,

                        "upload_timestamp": upload_ts,
                    }
                )

                chunk_count += 1

            results[file_name] = {
                "doc_id": doc_id,
                "chunks_created": chunk_count,
            }

    if not texts:

        logger.warning(
            "No non-empty chunks found to index"
        )

        return results

    # ------------------------------------------------------------------
    # Step 2: Generate embeddings
    # ------------------------------------------------------------------

    with log_time(
        f"embedding {len(texts)} chunk(s)"
    ):

        vectors = embeddings.embed_documents(
            texts
        )

    # ------------------------------------------------------------------
    # Step 3: Ensure Qdrant collection
    # ------------------------------------------------------------------

    ensure_collection(
        client=client,
        collection_name=collection_name,
        vector_size=len(vectors[0]),
    )

    # ------------------------------------------------------------------
    # Step 4: Create Qdrant points
    # ------------------------------------------------------------------

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=meta,
        )
        for vector, meta in zip(
            vectors,
            metadata,
        )
    ]

    # ------------------------------------------------------------------
    # Step 5: Remove previous version of the same documents
    # ------------------------------------------------------------------

    with log_time(
        "removing old chunks for re-uploaded files"
    ):

        for file_name in docs_by_file:

            old_count = delete_indexed_document(
                doc_id=compute_doc_id(file_name),
                client=client,
                collection_name=collection_name,
                session_id=session_id,
            )

            if old_count > 0:

                logger.info(
                    "Replaced %d old chunk(s) for '%s'",
                    old_count,
                    file_name,
                )

    # ------------------------------------------------------------------
    # Step 6: Store new vectors
    # ------------------------------------------------------------------

    with log_time(
        f"storing {len(points)} chunk(s) in Qdrant"
    ):

        client.upsert(
            collection_name=collection_name,
            points=points,
        )

    logger.info(
        "Stored %d chunk(s) in Qdrant",
        len(points),
    )

    return results


# ======================================================================
# Query
# ======================================================================

def query_with_references(
    question: str,
    top_k: int,
    client: QdrantClient,
    embeddings,
    llm: ChatOpenAI,
    collection_name: str,
    session_id: str,
) -> dict:
    """
    Retrieve chunks only from the current session and generate
    an answer using the LLM.
    """

    if not collection_exists(
        client,
        collection_name,
    ):

        return {
            "answer": (
                "No documents have been uploaded yet. "
                "Please upload files first."
            ),
            "references": [],
        }

    # ------------------------------------------------------------------
    # Step 1: Embed question
    # ------------------------------------------------------------------

    with log_time(
        "embedding the question"
    ):

        query_vector = embeddings.embed_query(
            question
        )

    # ------------------------------------------------------------------
    # Step 2: Create session filter
    # ------------------------------------------------------------------

    session_filter = Filter(
        must=[
            FieldCondition(
                key="session_id",
                match=MatchValue(
                    value=session_id
                ),
            )
        ]
    )

    # ------------------------------------------------------------------
    # Step 3: Search Qdrant
    # ------------------------------------------------------------------

    with log_time(
        "searching Qdrant"
    ):

        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=session_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

    if not results.points:

        return {
            "answer": (
                "No documents have been uploaded to "
                "this session yet."
            ),
            "references": [],
        }

    references: List[Reference] = []

    context_parts: List[str] = []

    # ------------------------------------------------------------------
    # Step 4: Build context and references
    # ------------------------------------------------------------------

    for hit in results.points:

        payload = hit.payload

        references.append(
            Reference(
                file_name=payload.get(
                    "file_name",
                    "Unknown",
                ),
                page_number=payload.get(
                    "page_number",
                    1,
                ),
            )
        )

        context_parts.append(
            f"[{payload.get('file_name', 'Unknown')} "
            f"- page {payload.get('page_number', '?')}]\n"
            f"{payload.get('chunk_text', '')}"
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    # ------------------------------------------------------------------
    # Step 5: Generate LLM answer
    # ------------------------------------------------------------------

    with log_time(
        "generating the answer"
    ):

        chain = (
            ANSWER_PROMPT
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

    return {
        "answer": answer,
        "references": references,
    }


# ======================================================================
# List documents
# ======================================================================

def list_indexed_documents(
    client: QdrantClient,
    collection_name: str,
    session_id: str,
) -> List[dict]:
    """
    Return documents belonging only to the supplied session.
    """

    if not collection_exists(
        client,
        collection_name,
    ):

        return []

    seen: Dict[str, dict] = {}

    offset = None

    session_filter = Filter(
        must=[
            FieldCondition(
                key="session_id",
                match=MatchValue(
                    value=session_id
                ),
            )
        ]
    )

    while True:

        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=session_filter,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:

            payload = record.payload

            doc_id = payload.get(
                "doc_id",
                "unknown",
            )

            if doc_id not in seen:

                seen[doc_id] = {
                    "doc_id": doc_id,

                    "file_name": payload.get(
                        "file_name",
                        "unknown",
                    ),

                    "file_type": payload.get(
                        "file_type",
                        "unknown",
                    ),

                    "upload_timestamp": payload.get(
                        "upload_timestamp",
                        "",
                    ),

                    "chunk_count": 0,
                }

            seen[doc_id]["chunk_count"] += 1

        if next_offset is None:
            break

        offset = next_offset

    return list(
        seen.values()
    )


# ======================================================================
# Delete one document
# ======================================================================

def delete_indexed_document(
    doc_id: str,
    client: QdrantClient,
    collection_name: str,
    session_id: str | None = None,
) -> int:
    """
    Delete document chunks.

    If session_id is supplied, deletion is restricted to that session.
    """

    if not collection_exists(
        client,
        collection_name,
    ):

        return 0

    conditions = [
        FieldCondition(
            key="doc_id",
            match=MatchValue(
                value=doc_id
            ),
        )
    ]

    if session_id is not None:

        conditions.append(
            FieldCondition(
                key="session_id",
                match=MatchValue(
                    value=session_id
                ),
            )
        )

    doc_filter = Filter(
        must=conditions
    )

    count = client.count(
        collection_name=collection_name,
        count_filter=doc_filter,
    ).count

    if count > 0:

        client.delete(
            collection_name=collection_name,
            points_selector=doc_filter,
        )

    return count


# ======================================================================
# Delete complete session from Qdrant
# ======================================================================

def delete_session_documents(
    session_id: str,
    client: QdrantClient,
    collection_name: str,
) -> int:
    """
    Delete every Qdrant chunk belonging to a session.
    """

    if not collection_exists(
        client,
        collection_name,
    ):

        return 0

    session_filter = Filter(
        must=[
            FieldCondition(
                key="session_id",
                match=MatchValue(
                    value=session_id
                ),
            )
        ]
    )

    count = client.count(
        collection_name=collection_name,
        count_filter=session_filter,
    ).count

    if count > 0:

        client.delete(
            collection_name=collection_name,
            points_selector=session_filter,
        )

    return count