"""
Application configuration — loads from .env via pydantic-settings.
All other modules import from here; never read os.environ directly.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# Resolve the backend root directory (where this file lives)
BACKEND_ROOT = Path(__file__).parent.resolve()


class Settings(BaseSettings):
    """
    Central settings object.  Values are read (in priority order) from:
      1. Real environment variables
      2. The .env file in the backend root
      3. The defaults declared below
    """

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # OpenAI
    # ------------------------------------------------------------------ #
    openai_api_key: str = Field(..., description="OpenAI API key (required)")
    llm_model: str = Field("gpt-4o-mini", description="Chat-completion model name")
    embedding_model: str = Field(
        "text-embedding-3-small", description="Embedding model name"
    )

    # ------------------------------------------------------------------ #
    # SQLite
    # ------------------------------------------------------------------ #
    database_path: str = Field(
        str(BACKEND_ROOT / "data" / "enterprise.db"),
        description="Absolute or relative path to the SQLite database file",
    )

    # ------------------------------------------------------------------ #
    # ChromaDB
    # ------------------------------------------------------------------ #
    chroma_persist_dir: str = Field(
        str(BACKEND_ROOT / "data" / "chroma_db"),
        description="Directory where ChromaDB persists its data",
    )
    chroma_collection_name: str = Field(
        "enterprise_docs",
        description="Name of the ChromaDB collection for document chunks",
    )

    # ------------------------------------------------------------------ #
    # API / Server
    # ------------------------------------------------------------------ #
    api_host: str = Field("0.0.0.0", description="Uvicorn bind host")
    api_port: int = Field(8000, description="Uvicorn bind port")
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173"],
        description="Allowed CORS origins (React dev server)",
    )

    # ------------------------------------------------------------------ #
    # Agent behaviour
    # ------------------------------------------------------------------ #
    vector_top_k: int = Field(3, description="Number of chunks to retrieve from ChromaDB")
    sql_max_rows: int = Field(50, description="Maximum rows returned from any SQL query")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def database_url(self) -> str:
        """SQLAlchemy-compatible SQLite URL."""
        return f"sqlite:///{self.database_path}"

    @property
    def database_dir(self) -> Path:
        return Path(self.database_path).parent

    @property
    def chroma_dir(self) -> Path:
        return Path(self.chroma_persist_dir)


# Module-level singleton — import this everywhere
settings = Settings()

# Ensure data directories exist at import time
settings.database_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_dir.mkdir(parents=True, exist_ok=True)
