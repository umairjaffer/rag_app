import requests
import streamlit as st


# ======================================================================
# Backend URLs
# ======================================================================

API_BASE_URL = "http://127.0.0.1:8000"

HEALTH_URL = f"{API_BASE_URL}/health"

SESSIONS_URL = f"{API_BASE_URL}/sessions"


# ======================================================================
# Page configuration
# ======================================================================

st.set_page_config(
    page_title="RAG Application",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================================
# Session state initialization
# ======================================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_upload_response" not in st.session_state:
    st.session_state.last_upload_response = None

if "delete_message" not in st.session_state:
    st.session_state.delete_message = None


# ======================================================================
# Helper functions
# ======================================================================

def check_backend():
    """Check whether FastAPI backend is available."""

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


def create_session():
    """Create a new backend session."""

    try:

        response = requests.post(
            SESSIONS_URL,
            timeout=20,
        )

        if response.status_code == 200:

            return response.json()

        st.error(
            f"Could not create session: "
            f"{response.text}"
        )

        return None

    except requests.RequestException as exc:

        st.error(
            f"Session creation failed: {exc}"
        )

        return None


def get_sessions():
    """Retrieve all sessions."""

    try:

        response = requests.get(
            SESSIONS_URL,
            timeout=20,
        )

        if response.status_code == 200:
            return response.json()

        return None

    except requests.RequestException:

        return None


def get_session_details(session_id):
    """Retrieve session details and persistent chat history."""

    try:

        response = requests.get(
            f"{SESSIONS_URL}/{session_id}",
            timeout=20,
        )

        if response.status_code == 200:
            return response.json()

        return None

    except requests.RequestException:

        return None


def delete_current_session(session_id):
    """Delete a complete session."""

    try:

        response = requests.delete(
            f"{SESSIONS_URL}/{session_id}",
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
            f"Session deletion failed. "
            f"HTTP {response.status_code}: {detail}"
        )

        return None

    except requests.RequestException as exc:

        st.error(
            f"Session deletion failed: {exc}"
        )

        return None


def upload_files(
    session_id,
    uploaded_files,
):
    """Upload one or multiple files to the current session."""

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
            f"{SESSIONS_URL}/{session_id}/upload",
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
    session_id,
    question,
    top_k,
):
    """Ask a question inside the current session."""

    payload = {
        "session_id": session_id,
        "question": question,
        "top_k": top_k,
    }

    try:

        response = requests.post(
            f"{SESSIONS_URL}/{session_id}/query",
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


def get_documents(session_id):
    """Retrieve documents for the current session."""

    try:

        response = requests.get(
            f"{SESSIONS_URL}/{session_id}/documents",
            timeout=20,
        )

        if response.status_code == 200:

            return response.json()

        return None

    except requests.RequestException:

        return None


def delete_document(
    session_id,
    doc_id,
):
    """Delete one document from the current session."""

    try:

        response = requests.delete(
            f"{SESSIONS_URL}/{session_id}/documents/{doc_id}",
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

    except requests.RequestException as exc:

        st.error(
            f"Delete request failed: {exc}"
        )

        return None


def display_sources(references):
    """
    Display only document name and page number.

    Retrieved chunk text is intentionally hidden.
    """

    if not references:
        return

    st.markdown(
        "<b>Sources</b>",
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

    st.caption(
        " | ".join(source_text)
    )


def load_chat_history(session_id):
    """Load persistent chat history from PostgreSQL."""

    details = get_session_details(
        session_id
    )

    if details is None:
        return []

    return details.get(
        "messages",
        [],
    )


# ======================================================================
# Basic UI styling
# ======================================================================

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

    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================================
# Header
# ======================================================================

st.title("RAG Application")

st.caption(
    "Upload documents, create sessions, and ask questions "
    "using session-specific indexed content."
)


# ======================================================================
# Backend health
# ======================================================================

health = check_backend()


# ======================================================================
# Sidebar
# ======================================================================

with st.sidebar:

    st.header("System")

    if health:

        st.success(
            "Backend connected"
        )

        st.caption(
            f"Embedding: "
            f"{health.get('embedding_provider', 'Unknown')}"
        )

        st.caption(
            f"LLM: "
            f"{health.get('llm_model', 'Unknown')}"
        )

        st.caption(
            "Vector store: Qdrant"
        )

        st.caption(
            "Database: PostgreSQL"
        )

        st.caption(
            f"Collection: "
            f"{health.get('collection', 'Unknown')}"
        )

    else:

        st.error(
            "Backend unavailable"
        )

        st.caption(
            "Start the FastAPI backend before "
            "using the application."
        )

    st.divider()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    st.header("Session Management")

    if st.button(
        "New Session",
        use_container_width=True,
        type="primary",
    ):

        session_response = create_session()

        if session_response:

            st.session_state.session_id = (
                session_response["session_id"]
            )

            st.session_state.chat_history = []

            st.session_state.last_upload_response = None

            st.rerun()

    # ------------------------------------------------------------------
    # Existing sessions
    # ------------------------------------------------------------------

    sessions_response = get_sessions()

    if sessions_response:

        sessions = sessions_response.get(
            "sessions",
            [],
        )

        if sessions:

            session_options = [
                session["session_id"]
                for session in sessions
            ]

            current_session = (
                st.session_state.session_id
            )

            if current_session not in session_options:

                current_session = session_options[0]

                st.session_state.session_id = (
                    current_session
                )

                st.session_state.chat_history = (
                    load_chat_history(
                        current_session
                    )
                )

            selected_session = st.selectbox(
                "Select session",
                session_options,
                index=session_options.index(
                    current_session
                ),
                format_func=lambda value: (
                    f"{value[:8]}..."
                ),
            )

            if selected_session != (
                st.session_state.session_id
            ):

                st.session_state.session_id = (
                    selected_session
                )

                st.session_state.chat_history = (
                    load_chat_history(
                        selected_session
                    )
                )

                st.rerun()

        else:

            st.info(
                "No sessions available. "
                "Create a new session."
            )

    # ------------------------------------------------------------------
    # Current session
    # ------------------------------------------------------------------

    if st.session_state.session_id:

        st.caption(
            f"Current session: "
            f"{st.session_state.session_id}"
        )

        if st.button(
            "Delete Current Session",
            use_container_width=True,
        ):

            with st.spinner(
                "Deleting session..."
            ):

                delete_result = (
                    delete_current_session(
                        st.session_state.session_id
                    )
                )

            if delete_result:

                deleted_session = (
                    st.session_state.session_id
                )

                st.session_state.session_id = None

                st.session_state.chat_history = []

                st.session_state.last_upload_response = None

                st.session_state.delete_message = (
                    f"Session {deleted_session[:8]} "
                    "and all associated data were deleted."
                )

                st.rerun()

    st.divider()

    # ------------------------------------------------------------------
    # Retrieval settings
    # ------------------------------------------------------------------

    st.header("Retrieval")

    top_k = st.slider(
        "Chunks to retrieve",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    st.divider()

    if st.button(
        "Clear Chat",
        use_container_width=True,
    ):

        st.session_state.chat_history = []

        st.rerun()


# ======================================================================
# Delete message
# ======================================================================

if st.session_state.delete_message:

    st.success(
        st.session_state.delete_message
    )

    st.session_state.delete_message = None


# ======================================================================
# Create initial session automatically
# ======================================================================

if (
    st.session_state.session_id is None
    and health
):

    session_response = create_session()

    if session_response:

        st.session_state.session_id = (
            session_response["session_id"]
        )

        st.session_state.chat_history = []

        st.rerun()


# ======================================================================
# Current session
# ======================================================================

current_session_id = (
    st.session_state.session_id
)


if current_session_id:

    st.subheader(
        f"Session: {current_session_id[:8]}..."
    )

    # ------------------------------------------------------------------
    # Document upload
    # ------------------------------------------------------------------

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
                    len(
                        uploaded_file.getvalue()
                    )
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
                "Uploading, processing, chunking, "
                "embedding, and indexing..."
            ):

                upload_response = upload_files(
                    current_session_id,
                    uploaded_files,
                )

            if upload_response:

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
                    if result.get("status")
                    == "indexed"
                ]

                failed_files = [
                    result
                    for result in results
                    if result.get("status")
                    == "error"
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

                col1, col2, col3 = st.columns(
                    3
                )

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
                        "Documents successfully indexed."
                    )

                    for result in successful_files:

                        st.write(
                            f"**{result.get('file_name')}** "
                            f"- "
                            f"{result.get('chunks_created', 0)} "
                            "chunks - Indexed"
                        )

                if failed_files:

                    st.error(
                        "Some files could not be indexed."
                    )

                    for result in failed_files:

                        st.write(
                            f"**{result.get('file_name')}**: "
                            f"{result.get('error', 'Unknown error')}"
                        )

    # ------------------------------------------------------------------
    # Indexed documents
    # ------------------------------------------------------------------

    st.divider()

    st.header("Indexed Documents")

    documents_response = get_documents(
        current_session_id
    )

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

                    st.caption(
                        "Uploaded"
                    )

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
                                    current_session_id,
                                    doc_id,
                                )
                            )

                        if delete_result:

                            st.session_state.delete_message = (
                                f"Successfully deleted "
                                f"{delete_result.get('file_name', file_name)}."
                            )

                            st.rerun()

                st.divider()

        else:

            st.info(
                "No documents are currently indexed "
                "in this session."
            )

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    st.header("Chat")

    st.caption(
        "Ask questions about the documents in this session."
    )

    # ------------------------------------------------------------------
    # Display persistent chat history
    # ------------------------------------------------------------------

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

                st.markdown(
                    content
                )

        elif role == "assistant":

            with st.chat_message("assistant"):

                st.markdown(
                    content
                )

                references = message.get(
                    "references",
                    [],
                )

                display_sources(
                    references
                )

    # ------------------------------------------------------------------
    # Chat input
    # ------------------------------------------------------------------

    question = st.chat_input(
        "Ask a question about your documents..."
    )

    if question:

        question = question.strip()

        if question:

            # ----------------------------------------------------------
            # Display user message immediately
            # ----------------------------------------------------------

            with st.chat_message("user"):

                st.markdown(
                    question
                )

            # ----------------------------------------------------------
            # Query backend
            # ----------------------------------------------------------

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching documents and generating answer..."
                ):

                    query_response = query_backend(
                        session_id=current_session_id,
                        question=question,
                        top_k=top_k,
                    )

                if query_response is None:

                    answer = (
                        "I could not get a response "
                        "from the backend."
                    )

                    references = []

                    st.error(
                        answer
                    )

                else:

                    answer = query_response.get(
                        "answer",
                        "No answer was generated.",
                    )

                    references = query_response.get(
                        "references",
                        [],
                    )

                    # Only LLM response is displayed.
                    st.markdown(
                        answer
                    )

                    # Only source metadata is displayed.
                    display_sources(
                        references
                    )

            # ----------------------------------------------------------
            # Save local state.
            #
            # The authoritative copy is also stored in PostgreSQL
            # by the backend.
            # ----------------------------------------------------------

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question,
                }
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

else:

    st.info(
        "Create a new session to start uploading documents."
    )