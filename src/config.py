"""Configuration management with Pydantic models for validation."""

import os
import json
from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="text-embedding-ada-002", description="Embedding model to use")
    batch_size: int = Field(default=20, ge=1, le=100, description="Batch size for embeddings")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    timeout: int = Field(default=30, ge=5, le=300, description="Request timeout in seconds")


class ChromaDBConfig(BaseModel):
    """ChromaDB configuration."""
    db_path: str = Field(default="./data/chroma_db", description="Path to ChromaDB storage")
    collection_name: str = Field(default="transcript_chunks", description="Collection name")
    distance_metric: Literal["cosine", "euclidean", "manhattan"] = Field(
        default="cosine", description="Distance metric for similarity search"
    )


class ChunkingConfig(BaseModel):
    """Chunking strategy configuration."""
    strategy: Literal["fixed_window", "speaker_based"] = Field(
        default="fixed_window", description="Chunking strategy"
    )
    window_size: int = Field(default=60, ge=10, le=600, description="Window size in seconds")
    overlap_seconds: int = Field(default=5, ge=0, le=30, description="Overlap between chunks")
    min_chunk_length: int = Field(default=10, ge=1, description="Minimum chunk length in characters")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Log level"
    )
    file_path: Optional[str] = Field(default="./logs/chunking_service.log", description="Log file path")
    max_file_size: int = Field(default=10_000_000, description="Max log file size in bytes")
    backup_count: int = Field(default=5, description="Number of backup log files")


class DevelopmentConfig(BaseModel):
    """Development-specific configuration."""
    environment: Literal["development", "production"] = Field(
        default="development", description="Environment mode"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    fast_mode: bool = Field(default=False, description="Skip embeddings for fast testing")
    cache_embeddings: bool = Field(default=True, description="Cache embeddings during development")


class AppConfig(BaseModel):
    """Main application configuration."""
    openai: OpenAIConfig
    chromadb: ChromaDBConfig
    chunking: ChunkingConfig
    logging: LoggingConfig
    development: DevelopmentConfig

    @validator("openai")
    def validate_openai_config(cls, v):
        """Validate OpenAI configuration."""
        if not v.api_key or v.api_key.startswith("your_"):
            raise ValueError("Valid OpenAI API key is required")
        return v

    @validator("chromadb")
    def validate_chromadb_config(cls, v):
        """Validate ChromaDB configuration."""
        # Ensure the directory exists
        Path(v.db_path).parent.mkdir(parents=True, exist_ok=True)
        return v


def load_environment_variables():
    """Load environment variables from .env file."""
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)


def create_config_from_env() -> AppConfig:
    """Create configuration from environment variables."""
    load_environment_variables()
    
    return AppConfig(
        openai=OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "text-embedding-ada-002"),
            batch_size=int(os.getenv("DEFAULT_BATCH_SIZE", "20")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
        ),
        chromadb=ChromaDBConfig(
            db_path=os.getenv("CHROMA_DB_PATH", "./data/chroma_db"),
            collection_name=os.getenv("CHROMA_COLLECTION", "transcript_chunks"),
        ),
        chunking=ChunkingConfig(
            window_size=int(os.getenv("DEFAULT_CHUNK_WINDOW", "60")),
        ),
        logging=LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            file_path=os.getenv("LOG_FILE", "./logs/chunking_service.log"),
        ),
        development=DevelopmentConfig(
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
        ),
    )


def load_config_from_file(config_path: str) -> AppConfig:
    """Load configuration from JSON file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, "r") as f:
        config_data = json.load(f)
    
    return AppConfig(**config_data)


def get_config(config_file: Optional[str] = None) -> AppConfig:
    """Get configuration with precedence: file > environment > defaults."""
    if config_file and Path(config_file).exists():
        return load_config_from_file(config_file)
    
    # Try to load local config file
    local_config = Path("config.local.json")
    if local_config.exists():
        return load_config_from_file(str(local_config))
    
    # Try to load default config file
    default_config = Path("config.default.json")
    if default_config.exists():
        base_config = load_config_from_file(str(default_config))
        # Override with environment variables
        load_environment_variables()
        if os.getenv("OPENAI_API_KEY"):
            base_config.openai.api_key = os.getenv("OPENAI_API_KEY")
        return base_config
    
    # Fall back to environment variables only
    return create_config_from_env()