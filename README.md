# RAG Application

A document-based RAG application built with **FastAPI, Streamlit, LangChain, Qdrant, PostgreSQL, Hugging Face/OpenAI Embeddings, and OpenAI LLM**.

## Features

* Single and multiple document upload
* PDF, DOCX, TXT, CSV, and Markdown support
* Document loading, metadata extraction, and chunking
* Hugging Face and OpenAI embedding support
* Qdrant vector storage and similarity search
* Configurable Top-K retrieval
* OpenAI LLM response generation
* Source references with document name and page number
* Session-based document isolation
* PostgreSQL session management
* Persistent chat history
* Create, list, and delete sessions
* Session-specific document listing and deletion
* Automatic deletion of session documents from Qdrant
* Upload and query error handling
* Backend operation timing and logging

## Project Structure

```text
rag_app/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── db_models.py
│   │   ├── file_uploader.py
│   │   ├── models.py
│   │   ├── rag_chain.py
│   │   ├── routes.py
│   │   └── timing.py
│   │
│   └── main.py
│
├── frontend/
│   └── streamlit_ui.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## RAG Flow

```text
Upload Files
     ↓
Load Documents
     ↓
Add Metadata
     ↓
Chunk Documents
     ↓
Generate Embeddings
     ↓
Store in Qdrant
     ↓
User Question
     ↓
Retrieve Top-K Chunks
     ↓
OpenAI LLM
     ↓
Answer + References
```

## Session Flow

```text
Create Session
     ↓
Upload Documents
     ↓
Documents linked to Session
     ↓
Ask Questions
     ↓
Chat History → PostgreSQL
     ↓
Answer + Sources
     ↓
Delete Session
     ↓
Qdrant Documents + Chat History Deleted
```

## Embeddings

### Hugging Face

```text
Model: sentence-transformers/all-MiniLM-L6-v2
Dimension: 384
```

### OpenAI

```text
Model: text-embedding-3-small
Dimension: 1536
```

Configured through:

```env
EMBEDDING_PROVIDER=huggingface
```

or:

```env
EMBEDDING_PROVIDER=openai
```

## Chunking

```text
RecursiveCharacterTextSplitter
Chunk Size: 500
Chunk Overlap: 50
```

## Storage

### Qdrant

Stores document vectors and metadata:

```text
session_id
doc_id
file_name
file_type
page_number
chunk_index
chunk_text
upload_timestamp
```

### PostgreSQL

Stores:

```text
sessions
chat_messages
```

## API Endpoints

```text
POST   /sessions
GET    /sessions
DELETE /sessions/{session_id}

POST   /upload
POST   /query
GET    /sessions/{session_id}/messages

GET    /documents
DELETE /documents/{doc_id}

GET    /health
```

## Run

### Backend

```bash
cd backend
python main.py
```

### Frontend

```bash
streamlit run frontend/streamlit_ui.py
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Environment Variables

```env
OPENAI_API_KEY=your_key
LLM_MODEL=gpt-4o-mini

EMBEDDING_PROVIDER=huggingface

QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=rag_uploads

RETRIEVER_K=5
MAX_UPLOAD_SIZE_MB=50

DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/rag_app
```

Do not commit `.env` or API keys to Git.
