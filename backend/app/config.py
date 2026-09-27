from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# rag_app/
ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    openai_api_key: str
    llm_model: str = "gpt-4o-mini"

    embedding_provider: str = "openai"

    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "rag_documents"

    retriever_k: int = 5

    upload_dir: str = "uploads"
    max_upload_size_mb: int = 20

    database_url: str

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "rag_app"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()