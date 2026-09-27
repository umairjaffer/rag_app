# RAG Application

A document-based Retrieval-Augmented Generation (RAG) application built with **FastAPI**, **Streamlit**, **LangChain**, **Qdrant**, **Hugging Face/OpenAI embeddings**, and **OpenAI Chat LLMs**.

The application allows users to upload multiple documents, process and chunk them, generate embeddings, store the vectors in Qdrant, and ask questions about the uploaded documents through a Streamlit chat interface.

## Features

- Multiple file upload
- Supported file formats:
  - PDF
  - DOCX
  - TXT
  - CSV
  - Markdown
- FastAPI backend
- Streamlit frontend
- Document loading with format-specific LangChain loaders
- Recursive character text splitting
- Hugging Face embeddings
- OpenAI embeddings
- Qdrant vector database
- Semantic similarity search
- Configurable Top-K retrieval
- LLM-generated answers using retrieved context
- Source attribution in generated responses
- File name and page number references
- Indexed document listing
- Document deletion
- Automatic deletion of all associated Qdrant chunks when a document is deleted
- Upload and indexing status
- Chunk count for each document
- Qdrant collection information
- Backend health check
- Chat history maintained in Streamlit session state
- Error handling for upload, query, connection, and deletion operations
- Environment-based configuration

---
## Project Structure

```text
rag-app/
│
├── backend/
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── config.py
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

---

# Backend

The backend is responsible for:

* File uploads
* Document loading
* Text chunking
* Embedding generation
* Qdrant indexing
* Vector search
* LLM response generation
* Document listing
* Document deletion
* Health monitoring

The backend is built using FastAPI.

---

## Backend Files

### `backend/main.py`

Main entry point for the FastAPI application.

Responsibilities:

* Create FastAPI application
* Configure CORS
* Initialize shared dependencies
* Load embedding model
* Initialize Qdrant client
* Initialize LLM
* Create upload directory
* Include API routes
* Clean up application state during shutdown

Run the backend with:

```bash
uvicorn backend.main:app --reload
```

---

### `backend/app/api.py`

Contains the main API handlers.

Responsibilities:

* Upload multiple files
* Query the RAG system
* List indexed documents
* Delete documents
* Health check

---

### `backend/app/file_uploader.py`

Responsible for loading supported document formats.

Supported loaders:

| File Type | Loader                       |
| --------- | ---------------------------- |
| PDF       | `PyPDFLoader`                |
| DOCX      | `Docx2txtLoader`             |
| Markdown  | `UnstructuredMarkdownLoader` |
| CSV       | `CSVLoader`                  |
| TXT       | `TextLoader`                 |

Markdown loading includes a fallback to `TextLoader` if the primary loader fails.

Each loaded document receives metadata such as:

```text
file_name
file_type
page_number
```

---

### `backend/app/rag_chain.py`

Contains the core RAG pipeline.

Responsibilities:

1. Create embeddings
2. Initialize Qdrant
3. Create/validate Qdrant collection
4. Split documents into chunks
5. Generate embeddings
6. Store vectors and metadata
7. Retrieve relevant chunks
8. Build context
9. Generate the final LLM response
10. List indexed documents
11. Delete indexed document chunks

---

### `backend/app/models.py`

Contains Pydantic request and response models.

Main models include:

```text
Reference
QueryRequest
QueryResponse
FileUploadResult
UploadResponse
DocumentInfo
DocumentListResponse
DeleteResponse
```

---

### `backend/app/routes.py`

Registers the backend API routes.

Current routes:

```text
POST   /upload
POST   /query
GET    /documents
DELETE /documents/{doc_id}
GET    /health
```

---

### `backend/app/config.py`

Contains application configuration using Pydantic Settings.

Configuration includes:

```text
OPENAI_API_KEY
LLM_MODEL
EMBEDDING_PROVIDER
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION
RETRIEVER_K
UPLOAD_DIR
MAX_UPLOAD_SIZE_MB
```

---

### `backend/app/timing.py`

Contains a small timing utility used to measure execution time of backend operations.

---

# Frontend

The frontend is built with Streamlit.

The frontend provides:

* Multiple file selection
* Upload and indexing interface
* Upload status
* File/chunk information
* Qdrant collection information
* Indexed document listing
* Document deletion
* Delete success messages
* Chat interface
* Source information
* Retrieval configuration
* Backend connection status

Run the frontend with:

```bash
streamlit run frontend/streamlit_ui.py
```

---

# RAG Pipeline

The application follows the following pipeline:

```text
Upload File
    |
    v
Validate File
    |
    v
Load Document
    |
    v
Add Metadata
    |
    v
Text Splitting
    |
    v
Generate Embeddings
    |
    v
Store Vectors + Metadata
    |
    v
Qdrant
```

For querying:

```text
User Question
      |
      v
Question Embedding
      |
      v
Qdrant Similarity Search
      |
      v
Top-K Relevant Chunks
      |
      v
Context Construction
      |
      v
OpenAI LLM
      |
      v
Generated Answer
      |
      v
Source Information
```

---

# Document Processing

## Supported Files

The application currently accepts:

```text
.pdf
.docx
.txt
.csv
.md
.markdown
```

The Streamlit uploader restricts file selection to these formats.

---

# Chunking

Documents are split using:

```python
RecursiveCharacterTextSplitter
```

Current configuration:

```text
Chunk size: 500
Chunk overlap: 50
```

The overlap helps preserve context between neighboring chunks.

Example:

```text
Document
   |
   +---- Chunk 1
   |
   +---- Chunk 2
   |
   +---- Chunk 3
```

---

# Embeddings

The application supports two embedding providers.

## Hugging Face

Default model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding dimension:

```text
384
```

Configured through:

```env
EMBEDDING_PROVIDER=huggingface
```

---

## OpenAI

Model:

```text
text-embedding-3-small
```

Embedding dimension:

```text
1536
```

Configured through:

```env
EMBEDDING_PROVIDER=openai
```

The required API key is:

```env
OPENAI_API_KEY=your_api_key
```

---

# Vector Database

The application uses **Qdrant** as the vector database.

Qdrant stores:

* Vector embeddings
* Document ID
* File name
* File type
* Page number
* Chunk index
* Chunk text
* Upload timestamp

Example payload:

```json
{
  "file_name": "example.pdf",
  "file_type": "pdf",
  "page_number": 1,
  "chunk_index": 0,
  "chunk_text": "Document content...",
  "doc_id": "a1b2c3d4e5f6g7h8",
  "upload_timestamp": "2026-09-26T10:00:00+00:00"
}
```

---

# Qdrant Collection

The collection name is configured using:

```env
QDRANT_COLLECTION=rag_uploads
```

The application uses:

```text
Distance: Cosine
```

The vector dimension must match the selected embedding model.

For example:

```text
Hugging Face
384 dimensions
        |
        v
Qdrant collection
384 dimensions
```

or:

```text
OpenAI
1536 dimensions
        |
        v
Qdrant collection
1536 dimensions
```

If the existing collection has a different vector dimension, the application raises a dimension mismatch error instead of inserting incompatible vectors.

---

# Document IDs

Each document receives a deterministic document ID based on its filename.

Conceptually:

```text
filename
   |
   v
SHA-256
   |
   v
Document ID
```

This ID is stored with every chunk belonging to that document.

The `doc_id` is then used for document-level operations such as deletion.

---

# Document Deletion

The application supports document deletion from the frontend.

The flow is:

```text
User clicks Delete
        |
        v
DELETE /documents/{doc_id}
        |
        v
Backend finds document chunks
        |
        v
Qdrant filter by doc_id
        |
        v
All chunks deleted
        |
        v
Delete response returned
        |
        v
Frontend shows success message
```

Example:

```text
Successfully deleted example.pdf and 25 chunks from Qdrant.
```

Deleting a document therefore removes its associated vectors/chunks from the Qdrant collection.

---

# Query and Retrieval

The user can configure the number of retrieved chunks using the Streamlit sidebar.

Default:

```text
Top-K = 5
```

Allowed range:

```text
1 - 20
```

For example:

```text
User Question
      |
      v
Embedding
      |
      v
Qdrant
      |
      +---- Chunk 1
      +---- Chunk 2
      +---- Chunk 3
      +---- Chunk 4
      +---- Chunk 5
      |
      v
LLM Context
```

---

# LLM Response

The LLM receives:

```text
User Question
+
Retrieved Context
```

The LLM then generates the final response.

The frontend displays the **LLM-generated answer as the main chat response**.

Raw retrieved chunk content is not displayed in the chat interface.

Source metadata can be shown separately:

```text
Sources

Source 1: example.pdf, Page 1
Source 2: example.pdf, Page 3
```

The intended response format can also include source attribution in the generated answer, for example:

```text
The main objective is to develop a system for ...
(Source: example.pdf, Page 1)
```

The exact source wording depends on the backend prompt configuration.

---

# Frontend Chat

The Streamlit chat interface maintains chat history using:

```python
st.session_state.chat_history
```

A typical conversation looks like:

```text
User:
What is the main objective?

Assistant:
The main objective is ...

(Source: example.pdf, Page 1)

Sources:
Source 1: example.pdf, Page 1
Source 2: example.pdf, Page 2
```

Retrieved chunk text itself is not rendered as the main answer.

---

# API Documentation

FastAPI automatically provides interactive API documentation.

After starting the backend, open:

```text
http://127.0.0.1:8000/docs
```

Alternative documentation:

```text
http://127.0.0.1:8000/redoc
```

---

# API Endpoints

## Upload Documents

```http
POST /upload
```

Uploads one or more files.

Request:

```text
multipart/form-data
```

Field:

```text
files
```

Multiple files can be sent using the same field name.

Example response:

```json
{
  "total_files": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "file_name": "document1.pdf",
      "doc_id": "abc123",
      "file_type": "pdf",
      "status": "indexed",
      "chunks_created": 18,
      "error": null
    },
    {
      "file_name": "document2.txt",
      "doc_id": "def456",
      "file_type": "txt",
      "status": "indexed",
      "chunks_created": 7,
      "error": null
    }
  ]
}
```

---

## Query Documents

```http
POST /query
```

Request:

```json
{
  "question": "What is the main objective?",
  "top_k": 5
}
```

Response:

```json
{
  "question": "What is the main objective?",
  "answer": "The main objective is ...",
  "references": [
    {
      "file_name": "example.pdf",
      "file_type": "pdf",
      "page_number": 1,
      "chunk_index": 2,
      "chunk_text": "Retrieved context...",
      "relevance_score": 0.87
    }
  ],
  "embedding_provider": "huggingface"
}
```

The backend returns the retrieved chunk text as API reference data so the application can maintain traceability.

The Streamlit frontend does not display that raw chunk content.

---

## List Documents

```http
GET /documents
```

Returns currently indexed documents.

Example:

```json
{
  "documents": [
    {
      "doc_id": "abc123",
      "file_name": "example.pdf",
      "file_type": "pdf",
      "chunk_count": 20,
      "upload_timestamp": "2026-09-26T10:00:00+00:00"
    }
  ],
  "total": 1
}
```

---

## Delete Document

```http
DELETE /documents/{doc_id}
```

Deletes all Qdrant chunks associated with the specified document.

Example:

```http
DELETE /documents/abc123
```

Response:

```json
{
  "status": "deleted",
  "doc_id": "abc123",
  "file_name": "example.pdf",
  "chunks_deleted": 20
}
```

---

## Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "embedding_provider": "huggingface",
  "llm_model": "gpt-4o-mini",
  "collection": "rag_uploads"
}
```

---

# Environment Variables

Create a `.env` file in the project root.

Example:

```env
OPENAI_API_KEY=your_openai_api_key

LLM_MODEL=gpt-4o-mini

EMBEDDING_PROVIDER=huggingface

QDRANT_URL=https://your-qdrant-instance-url

QDRANT_API_KEY=your_qdrant_api_key

QDRANT_COLLECTION=rag_uploads

RETRIEVER_K=5

UPLOAD_DIR=upload_tmp

MAX_UPLOAD_SIZE_MB=50
```

Do not commit `.env` to Git.

---

# Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd rag-app
```

---

## 2. Create a Virtual Environment

Using Conda:

```bash
conda create -n rag-app python=3.11
```

Activate it:

```bash
conda activate rag-app
```

Or using Python virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

The application has two separate processes:

```text
FastAPI Backend
       +
Streamlit Frontend
```

## Start Backend

From the project root:

```bash
uvicorn backend.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Start Frontend

Open another terminal and activate the same environment.

Then run:

```bash
streamlit run frontend/streamlit_ui.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

---

# Complete Workflow

## Step 1: Start Backend

```bash
uvicorn backend.main:app --reload
```

---

## Step 2: Start Frontend

```bash
streamlit run frontend/streamlit_ui.py
```

---

## Step 3: Upload Documents

Select one or more supported files.

Example:

```text
research-paper.pdf
notes.docx
data.csv
readme.md
information.txt
```

Click:

```text
Upload and Index
```

The backend will:

```text
Validate
   ↓
Load
   ↓
Split
   ↓
Embed
   ↓
Index
   ↓
Qdrant
```

---

## Step 4: Verify Indexing

The frontend shows:

```text
Files
Indexed
Failed
```

and the number of chunks created for each document.

The indexed documents section also shows:

```text
File
Type
Chunk count
Upload time
Delete
```

---

## Step 5: Ask Questions

Enter a question in the chat.

Example:

```text
What is the main objective of this research?
```

The backend:

```text
Question
   ↓
Embedding
   ↓
Qdrant Retrieval
   ↓
Relevant Context
   ↓
LLM
   ↓
Answer
```

The frontend displays the generated answer and source information.

---

## Step 6: Delete a Document

Click:

```text
Delete
```

for the required document.

The backend deletes the document's associated chunks from Qdrant.

The frontend then displays a success message containing the deleted file and number of deleted chunks.

---

# Error Handling

The application handles common errors including:

* Unsupported file types
* Empty uploads
* File size limits
* Upload failures
* Backend connection failures
* Query failures
* Query timeouts
* Delete failures
* Qdrant connection issues
* Embedding dimension mismatch
* Missing documents
* Missing Qdrant collection

---

# Configuration

## Maximum File Size

Configured using:

```env
MAX_UPLOAD_SIZE_MB=50
```

---

## Chunk Configuration

Currently defined in:

```text
backend/app/rag_chain.py
```

```python
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
```

---

## Retrieval Configuration

The default retrieval value is:

```env
RETRIEVER_K=5
```

The Streamlit frontend also allows the user to dynamically select Top-K from:

```text
1 - 20
```

---

# Security Considerations

The current project is designed as a development/learning RAG application.

Before production deployment, consider adding:

* Authentication
* Authorization
* User-specific document isolation
* Restricted CORS origins
* Rate limiting
* File content validation
* Stronger file size controls
* API authentication
* Secure secret management
* HTTPS
* Request logging
* Monitoring
* Input validation
* Malware scanning for uploaded files
* Per-user Qdrant filtering
* Production-grade error handling

The current CORS configuration is permissive for local development.

---

# Current Limitations

## 1. Real-Time Backend Progress

The frontend displays processing status while waiting for the upload request, but the current backend returns the final upload response only after processing completes.

Therefore, true backend stage-by-stage progress such as:

```text
Loading PDF...
Chunking...
Generating embeddings...
Uploading to Qdrant...
```

is not streamed live from the backend.

True real-time progress would require:

* Server-Sent Events (SSE)
* WebSockets
* Background jobs with progress tracking
* Another asynchronous progress mechanism

---

## 2. Chat History

The Streamlit frontend maintains chat history visually.

However, the current `/query` request sends the current question to the backend.

Therefore, previous messages are not automatically sent to the LLM as conversational context.

For example:

```text
User:
What is the first method?

Assistant:
...

User:
What about the second one?
```

The backend may not understand what "the second one" refers to unless conversation history is explicitly added to the query pipeline.

---

## 3. Document ID Based on Filename

The current document ID is derived from the filename.

Therefore, two different documents with the exact same filename can map to the same document ID.

For a production application, a UUID or content-based document identifier would be more appropriate.

---

## 4. PDF Visual Extraction

The current PDF pipeline uses text extraction through `PyPDFLoader`.

PDF diagrams, images, and complex visual layouts may not be represented correctly in the extracted text.

A multimodal or dedicated document-processing pipeline would be required for reliable visual/table extraction.

---

# Technology Stack

| Component       | Technology            |
| --------------- | --------------------- |
| Frontend        | Streamlit             |
| Backend         | FastAPI               |
| RAG Framework   | LangChain             |
| LLM             | OpenAI                |
| Embeddings      | Hugging Face / OpenAI |
| Vector Database | Qdrant                |
| Validation      | Pydantic              |
| HTTP Client     | Requests              |
| Language        | Python                |

---

# Main Dependencies

Typical dependencies include:

```text
fastapi
uvicorn
streamlit
requests
langchain
langchain-openai
langchain-huggingface
langchain-community
langchain-text-splitters
qdrant-client
pydantic
pydantic-settings
python-dotenv
pypdf
docx2txt
unstructured
```

The exact installed versions should be maintained in:

```text
requirements.txt
```

---

# Development Workflow

A typical development workflow is:

```text
1. Start Qdrant
       ↓
2. Configure .env
       ↓
3. Start FastAPI
       ↓
4. Verify /health
       ↓
5. Open /docs
       ↓
6. Start Streamlit
       ↓
7. Upload documents
       ↓
8. Verify Qdrant indexing
       ↓
9. Ask questions
       ↓
10. Verify source references
       ↓
11. Test document deletion
```

---

# Future Improvements

Potential future improvements include:

* Streaming LLM responses
* Real-time upload progress using SSE/WebSockets
* Conversational RAG
* User authentication
* Multi-user document isolation
* Hybrid search
* BM25 + dense retrieval
* Reranking
* Metadata filtering
* Advanced retrievers
* Query rewriting
* Multi-query retrieval
* Parent-document retrieval
* Context compression
* RAG evaluation
* LangSmith tracing
* Langfuse observability
* Background document processing
* Redis/Celery task queue
* PostgreSQL metadata database
* Docker deployment
* CI/CD
* Cloud deployment
* Production monitoring

---

# License

Add your preferred license here.

Example:

```text
MIT License
```

---

# Author

Muhammad Umair

AI Engineer | Full Stack AI Developer

Focus areas:

* Machine Learning
* Deep Learning
* Generative AI
* RAG
* Agentic AI
* Computer Vision
* NLP
* FastAPI
* Cloud Deployment
* MLOps

```