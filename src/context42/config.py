"""Configuration settings for Context42."""

from pathlib import Path
from pydantic_settings import BaseSettings
from platformdirs import user_data_dir


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Paths
    data_dir: Path = Path(user_data_dir("context42"))

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Chunking (CHARACTERS, not tokens)
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Batch processing
    batch_size: int = 50  # Chunks per batch for embedding/insert

    # Search
    default_top_k: int = 5

    class Config:
        """Pydantic configuration."""
        env_prefix = "C42_"

    @property
    def db_path(self) -> Path:
        """Path to SQLite database."""
        return self.data_dir / "context42.db"

    @property
    def vectors_path(self) -> Path:
        """Path to LanceDB vector storage."""
        return self.data_dir / "vectors"

    @property
    def models_path(self) -> Path:
        """Path to model cache directory."""
        return self.data_dir / "models"

    def ensure_dirs(self) -> None:
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
