"""Application settings, loaded automatically from the .env file.

All configuration lives in one place. Every other module imports the
`settings` object from here instead of reading environment variables
directly.

Each field below is linked to its matching variable in .env through
`alias=`. This mapping must match your .env file exactly (the aliases
are case-sensitive), for example OPENAI_API_KEY -> openai_api_key.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application settings."""

    # --- OpenAI -------------------------------------------------------
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    llm_model: str = Field("gpt-4o-mini", alias="LLM_MODEL")

    # --- Embeddings -----------------------------------------------------
    # "huggingface" = free, runs locally, 384-dimensional vectors
    # "openai"      = paid, cloud API, 1536-dimensional vectors
    # NOTE: once documents are indexed with one provider, switching
    # providers requires deleting the Qdrant collection and re-indexing,
    # because the vector sizes are different.
    embedding_provider: str = Field("huggingface", alias="EMBEDDING_PROVIDER")

    # --- Qdrant -----------------------------------------------------------
    qdrant_url: str = Field(..., alias="QDRANT_URL")
    qdrant_api_key: str = Field(..., alias="QDRANT_API_KEY")
    qdrant_collection: str = Field("rag_uploads", alias="QDRANT_COLLECTION")

    # --- Retrieval --------------------------------------------------------
    retriever_k: int = Field(5, alias="RETRIEVER_K")  # Default chunks per query

    # --- File upload --------------------------------------------------
    upload_dir: str = Field("upload_tmp", alias="UPLOAD_DIR")  # Temp folder for uploads
    max_upload_size_mb: int = Field(50, alias="MAX_UPLOAD_SIZE_MB")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Silently skip unknown env vars
        populate_by_name=True,
    )


# One shared instance -- every other module imports this object.
settings = Settings()
