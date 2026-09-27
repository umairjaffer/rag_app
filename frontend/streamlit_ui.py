import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"

UPLOAD_URL = f"{API_BASE_URL}/upload"
QUERY_URL = f"{API_BASE_URL}/query"
HEALTH_URL = f"{API_BASE_URL}/health"
DOCUMENTS_URL = f"{API_BASE_URL}/documents"


st.set_page_config(
    page_title="RAG Application",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Session State
# -----------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_upload_response" not in st.session_state:
    st.session_state.last_upload_response = None

if "delete_message" not in st.session_state:
    st.session_state.delete_message = None


# -----------------------------
# Helper Functions
# -----------------------------

def check_backend():
    try:
        response = requests.get(
            HEALTH_URL,
            timeout=10,
        )

        if response.status_code == 200:
            return response.json()

        return None

    except requests.RequestException:
        return None


def upload_files(uploaded_files):

    files = []

    for uploaded_file in uploaded_files:
        files.append(
            (
                "files",
                (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                    or "application/octet-stream",
                ),
            )
        )

    try:
        response = requests.post(
            UPLOAD_URL,
            files=files,
            timeout=600,
        )

        if response.status_code == 200:
            return response.json()

        try:
            detail = response.json().get(
                "detail",
                response.text,
            )
        except ValueError:
            detail = response.text

        st.error(
            f"Upload failed. "
            f"HTTP {response.status_code}: {detail}"
        )

        return None

    except requests.Timeout:
        st.error(
            "The upload request timed out."
        )
        return None

    except requests.ConnectionError:
        st.error(
            "Could not connect to the FastAPI backend."
        )
        return None

    except requests.RequestException as exc:
        st.error(
            f"Upload request failed: {exc}"
        )
        return None


def query_backend(
    question: str,
    top_k: int,
):

    payload = {
        "question": question,
        "top_k": top_k,
    }

    try:
        response = requests.post(
            QUERY_URL,
            json=payload,
            timeout=180,
        )

        if response.status_code == 200:
            return response.json()

        try:
            detail = response.json().get(
                "detail",
                response.text,
            )
        except ValueError:
            detail = response.text

        st.error(
            f"Query failed. "
            f"HTTP {response.status_code}: {detail}"
        )

        return None

    except requests.Timeout:
        st.error(
            "The query timed out. Please try again."
        )
        return None

    except requests.ConnectionError:
        st.error(
            "Could not connect to the FastAPI backend."
        )
        return None

    except requests.RequestException as exc:
        st.error(
            f"Query request failed: {exc}"
        )
        return None


def get_documents():

    try:
        response = requests.get(
            DOCUMENTS_URL,
            timeout=20,
        )

        if response.status_code == 200:
            return response.json()

        return None

    except requests.RequestException:
        return None


def delete_document(doc_id):

    try:
        response = requests.delete(
            f"{DOCUMENTS_URL}/{doc_id}",
            timeout=60,
        )

        if response.status_code == 200:
            return response.json()

        try:
            detail = response.json().get(
                "detail",
                response.text,
            )
        except ValueError:
            detail = response.text

        st.error(
            f"Delete failed. "
            f"HTTP {response.status_code}: {detail}"
        )

        return None

    except requests.Timeout:
        st.error(
            "The delete request timed out."
        )
        return None

    except requests.ConnectionError:
        st.error(
            "Could not connect to the FastAPI backend."
        )
        return None

    except requests.RequestException as exc:
        st.error(
            f"Delete request failed: {exc}"
        )
        return None


def clear_chat():
    st.session_state.chat_history = []


def display_sources(references):

    if not references:
        return

    st.markdown(
        "<div style='margin-top: 8px; margin-bottom: 4px;'>"
        "<b>Sources</b>"
        "</div>",
        unsafe_allow_html=True,
    )

    source_text = []

    for index, reference in enumerate(
        references,
        start=1,
    ):

        file_name = reference.get(
            "file_name",
            "Unknown file",
        )

        page_number = reference.get(
            "page_number",
            "Unknown",
        )

        source_text.append(
            f"Source {index}: "
            f"{file_name}, "
            f"Page {page_number}"
        )

    st.caption("  |  ".join(source_text))


# -----------------------------
# Custom UI Styling
# -----------------------------

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1100px;
    }

    h1 {
        margin-bottom: 0.2rem;
    }

    h2 {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    h3 {
        margin-top: 0.7rem;
        margin-bottom: 0.4rem;
    }

    [data-testid="stChatMessage"] {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }

    [data-testid="stVerticalBlock"] {
        gap: 0.6rem;
    }

    .source-box {
        padding: 6px 10px;
        border-radius: 6px;
        margin-top: 4px;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Header
# -----------------------------

st.title("📚 RAG Application")

st.caption(
    "Upload documents, index them in Qdrant, "
    "and ask questions using the indexed content."
)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header("System")

    health = check_backend()

    if health:

        st.success("Backend connected")

        st.caption(
            f"Embedding: "
            f"{health.get('embedding_provider', 'Unknown')}"
        )

        st.caption(
            f"LLM: "
            f"{health.get('llm_model', 'Unknown')}"
        )

        st.caption("Vector store: Qdrant")

        st.caption(
            f"Collection: "
            f"{health.get('collection', 'Unknown')}"
        )

    else:

        st.error("Backend unavailable")

        st.caption(
            "Start the FastAPI backend before "
            "using the application."
        )

    st.divider()

    st.header("Retrieval")

    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help=(
            "Number of relevant chunks retrieved "
            "for each question."
        ),
    )

    st.divider()

    if st.button(
        "Clear Chat",
        use_container_width=True,
    ):

        clear_chat()

        st.rerun()


# -----------------------------
# Document Upload
# -----------------------------

st.header("Document Upload")

st.caption(
    "Supported: PDF, DOCX, Markdown, CSV, TXT"
)


uploaded_files = st.file_uploader(
    "Select one or more files",
    type=[
        "pdf",
        "docx",
        "md",
        "markdown",
        "csv",
        "txt",
    ],
    accept_multiple_files=True,
)


if uploaded_files:

    st.caption(
        f"{len(uploaded_files)} file(s) selected"
    )

    with st.expander(
        "View selected files",
        expanded=False,
    ):

        for uploaded_file in uploaded_files:

            file_size_mb = (
                len(uploaded_file.getvalue())
                / (1024 * 1024)
            )

            st.write(
                f"{uploaded_file.name} "
                f"({file_size_mb:.2f} MB)"
            )

    if st.button(
        "Upload and Index",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Uploading, processing, chunking, embedding, and indexing..."
        ):

            upload_response = upload_files(
                uploaded_files
            )

        if upload_response is not None:

            st.session_state.last_upload_response = (
                upload_response
            )

            results = upload_response.get(
                "results",
                [],
            )

            successful_files = [
                result
                for result in results
                if result.get("status") == "indexed"
            ]

            failed_files = [
                result
                for result in results
                if result.get("status") == "error"
            ]

            total_files = upload_response.get(
                "total_files",
                0,
            )

            successful = upload_response.get(
                "successful",
                0,
            )

            failed = upload_response.get(
                "failed",
                0,
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Files",
                    total_files,
                )

            with col2:
                st.metric(
                    "Indexed",
                    successful,
                )

            with col3:
                st.metric(
                    "Failed",
                    failed,
                )

            if successful_files:

                st.success(
                    "Documents successfully indexed in Qdrant."
                )

                for result in successful_files:

                    file_name = result.get(
                        "file_name",
                        "Unknown file",
                    )

                    chunks_created = result.get(
                        "chunks_created",
                        0,
                    )

                    st.write(
                        f"**{file_name}** — "
                        f"{chunks_created} chunks — "
                        f"Indexed"
                    )

            if failed_files:

                st.error(
                    "Some files could not be indexed."
                )

                for result in failed_files:

                    file_name = result.get(
                        "file_name",
                        "Unknown file",
                    )

                    error = result.get(
                        "error",
                        "Unknown error",
                    )

                    st.write(
                        f"**{file_name}**: {error}"
                    )


# -----------------------------
# Indexed Documents
# -----------------------------

st.divider()

st.header("Indexed Documents")


documents_response = get_documents()


if documents_response is None:

    st.warning(
        "Could not retrieve indexed documents."
    )

else:

    documents = documents_response.get(
        "documents",
        [],
    )

    total_documents = documents_response.get(
        "total",
        0,
    )

    if documents:

        st.caption(
            f"{total_documents} document(s) indexed"
        )

        for document in documents:

            file_name = document.get(
                "file_name",
                "Unknown",
            )

            file_type = document.get(
                "file_type",
                "Unknown",
            )

            chunk_count = document.get(
                "chunk_count",
                0,
            )

            upload_timestamp = document.get(
                "upload_timestamp",
                "",
            )

            doc_id = document.get(
                "doc_id",
                "",
            )

            col1, col2, col3, col4 = st.columns(
                [4, 1.5, 2.5, 1]
            )

            with col1:

                st.write(
                    f"**{file_name}**"
                )

                st.caption(
                    f"{file_type.upper()} | "
                    f"{chunk_count} chunks"
                )

            with col2:

                st.metric(
                    "Chunks",
                    chunk_count,
                )

            with col3:

                st.caption("Uploaded")

                st.caption(
                    upload_timestamp
                )

            with col4:

                if st.button(
                    "Delete",
                    key=f"delete_{doc_id}",
                    use_container_width=True,
                ):

                    with st.spinner(
                        "Deleting..."
                    ):

                        delete_result = (
                            delete_document(
                                doc_id
                            )
                        )

                    if delete_result:

                        deleted_file = (
                            delete_result.get(
                                "file_name",
                                file_name,
                            )
                        )

                        deleted_chunks = (
                            delete_result.get(
                                "chunks_deleted",
                                chunk_count,
                            )
                        )

                        st.session_state.delete_message = (
                            f"Successfully deleted "
                            f"{deleted_file} and "
                            f"{deleted_chunks} chunks "
                            f"from Qdrant."
                        )

                        st.rerun()

            st.divider()

    else:

        st.info(
            "No documents are currently indexed."
        )


if st.session_state.delete_message:

    st.success(
        st.session_state.delete_message
    )

    st.session_state.delete_message = None


# -----------------------------
# Chat
# -----------------------------

st.header("Chat")

st.caption(
    "Ask questions about your indexed documents."
)


# Display previous conversation
for message in st.session_state.chat_history:

    role = message.get(
        "role"
    )

    content = message.get(
        "content",
        "",
    )

    if role == "user":

        with st.chat_message("user"):

            st.markdown(content)

    elif role == "assistant":

        with st.chat_message("assistant"):

            # Only LLM-generated response
            st.markdown(content)

            references = message.get(
                "references",
                [],
            )

            # Source metadata only.
            # Chunk content is intentionally hidden.
            if references:
                display_sources(
                    references
                )


question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    question = question.strip()

    if question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                query_response = query_backend(
                    question=question,
                    top_k=top_k,
                )

            if query_response is None:

                answer = (
                    "I could not get a response "
                    "from the backend."
                )

                references = []

                st.error(answer)

            else:

                answer = query_response.get(
                    "answer",
                    "No answer was generated.",
                )

                references = query_response.get(
                    "references",
                    [],
                )

                # Main content:
                # ONLY the LLM-generated answer.
                st.markdown(answer)

                # Sources:
                # metadata only, no chunk content.
                display_sources(
                    references
                )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "references": references,
                }
            )

    else:

        st.warning(
            "Please enter a question."
        )