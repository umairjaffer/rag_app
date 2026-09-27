# RAG Application

A document-based Retrieval-Augmented Generation (RAG) application built with **FastAPI, Streamlit, LangChain, Qdrant, PostgreSQL, OpenAI Embeddings, OpenAI LLM, and LangSmith**.

## Features

* Single and multiple document upload
* PDF, DOCX, TXT, CSV, and Markdown support
* Document loading and metadata extraction
* Recursive character text splitting
* OpenAI embedding support
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
* LangSmith tracing for document loading, embedding, retrieval, and LLM operations

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
Validate Files
     ↓
Load Documents
     ↓
Add Metadata
     ↓
Chunk Documents
     ↓
Generate OpenAI Embeddings
     ↓
Store Vectors + Metadata in Qdrant
     ↓
User Question
     ↓
Generate Query Embedding
     ↓
Retrieve Top-K Chunks from Qdrant
     ↓
Build Context
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
Documents Linked to Session
     ↓
Documents Indexed in Qdrant
     ↓
Ask Questions
     ↓
Retrieve Session-Specific Chunks
     ↓
Generate Answer
     ↓
Save Chat History + Sources in PostgreSQL
     ↓
Delete Session
     ↓
Qdrant Documents + PostgreSQL Session Data Deleted
```

## LangSmith Tracing

LangSmith is used for tracing and observability of the RAG pipeline.

### Upload Tracing

```text
Upload Files API
└── RAG Document Indexing
    ├── Load File
    │   └── Select Document Loader
    └── Embed Documents
```

### Query Tracing

```text
Query API
└── RAG Query
    ├── Embed Question
    ├── Qdrant Retrieval
    └── ChatOpenAI
```

LangSmith tracing helps inspect:

* Document loading
* Document embedding
* Query embedding
* Qdrant retrieval
* Retrieved context
* LLM execution
* Execution time
* Errors during traced operations

## Supported Document Loaders

| File Type         | Loader                     |
| ----------------- | -------------------------- |
| PDF               | PyPDFLoader                |
| DOCX              | Docx2txtLoader             |
| Markdown          | UnstructuredMarkdownLoader |
| Markdown Fallback | TextLoader                 |
| CSV               | CSVLoader                  |
| TXT               | TextLoader                 |

## Embeddings

The application uses **OpenAI embeddings**.

```text
Model: text-embedding-3-small
Dimension: 1536
```

Configured through:

```env
EMBEDDING_PROVIDER=openai
```

## Chunking

The application uses `RecursiveCharacterTextSplitter`.

```text
Chunk Size: 800
Chunk Overlap: 100
```

## Storage

### Qdrant

The application uses a **remote Qdrant instance** for vector storage.

Qdrant stores document vectors and metadata.

Payload includes:

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

The application uses `session_id` and `doc_id` for document isolation and management.

### PostgreSQL

PostgreSQL stores application-level data:

```text
sessions
documents
chat_messages
message_sources
processing_logs
```

## API Endpoints

```text
GET    /sessions
POST   /sessions

GET    /sessions/{session_id}
DELETE /sessions/{session_id}

POST   /sessions/{session_id}/upload

GET    /sessions/{session_id}/documents
DELETE /sessions/{session_id}/documents/{doc_id}

POST   /sessions/{session_id}/query

GET    /health
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/umairjaffer/rag_app.git

cd rag_app
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure the `langsmith` package is installed because LangSmith tracing is used by the application.

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key

LLM_MODEL=gpt-4o-mini

EMBEDDING_PROVIDER=openai

QDRANT_URL=your_remote_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=rag_documents

RETRIEVER_K=5

MAX_UPLOAD_SIZE_MB=20

DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/rag_app

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=rag_app
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Do not commit `.env` or API keys to Git.

### 5. Set Up PostgreSQL

Make sure PostgreSQL is installed and running.

Create the database:

```bash
psql -U postgres -c "CREATE DATABASE rag_app;"
```

Update `DATABASE_URL` according to your PostgreSQL username, password, host, port, and database name.

The application automatically creates the required tables when the backend starts.

### 6. Configure Remote Qdrant

This application uses a **remote Qdrant instance** instead of a local Docker-based Qdrant server.

Add your remote Qdrant credentials to `.env`:

```env
QDRANT_URL=your_remote_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=rag_documents
```

The backend connects to the remote Qdrant instance during startup and creates the collection automatically if it does not already exist.

### 7. Configure LangSmith

Create a LangSmith API key and add the following variables to `.env`:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=rag_app
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

When tracing is enabled, the application records important RAG operations such as:

```text
Document Loading
      ↓
Document Embedding
      ↓
Qdrant Retrieval
      ↓
LLM Generation
```

These traces can be inspected in the configured LangSmith project.

### 8. Run the Backend

From the project root:

```bash
cd backend

python main.py
```

The FastAPI server runs on:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 9. Run the Frontend

Open a new terminal with the virtual environment activated:

```bash
streamlit run frontend/streamlit_ui.py
```

### 10. Verify the Application

1. Open the Streamlit interface.
2. Create a session.
3. Upload one or multiple supported documents.
4. Confirm that the documents are indexed successfully.
5. Ask a question about the uploaded documents.
6. Confirm that the answer contains source references.
7. Open LangSmith and verify the upload/query traces.

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

### API Documentation

```text
http://127.0.0.1:8000/docs
```

## Environment Variables

```env
OPENAI_API_KEY=your_openai_api_key

LLM_MODEL=gpt-4o-mini

EMBEDDING_PROVIDER=openai

QDRANT_URL=your_remote_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=rag_documents

RETRIEVER_K=5

MAX_UPLOAD_SIZE_MB=20

DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/rag_app

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=rag_app
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

## Security

* Never commit `.env` to Git.
* Never expose OpenAI API keys.
* Never expose Qdrant API keys.
* Never expose LangSmith API keys.
* Keep database credentials private.
