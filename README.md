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

## Getting Started (Clone & Setup)

Follow these steps to clone the repository and make it fully workable on your machine.

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

### 4. Configure Environment Variables

Copy the example file (or create a new one) and fill in your own values:

```bash
cp .env.example .env
```

Then edit `.env` and set:

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

### 5. Set Up PostgreSQL

Make sure PostgreSQL is installed and running, then create the database:

```bash
psql -U postgres -c "CREATE DATABASE rag_app;"
```

Update `DATABASE_URL` in `.env` to match your PostgreSQL username, password, host, and port.

### 6. Set Up Qdrant

* Use a local Qdrant instance (Docker) or a hosted Qdrant Cloud instance.

```bash
docker run -p 6333:6333 qdrant/qdrant
```

* Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env` accordingly.

### 7. Run the Backend

```bash
cd backend
python main.py
```

Verify it's working by visiting:

```text
http://127.0.0.1:8000/docs
```

### 8. Run the Frontend

In a new terminal (with the virtual environment activated):

```bash
streamlit run frontend/streamlit_ui.py
```

### 9. Verify Everything Works

* Open the Streamlit UI in your browser
* Create a session
* Upload a document
* Ask a question and confirm you get an answer with source references

If all these steps succeed, the application is fully cloned and workable.

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