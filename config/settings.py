"""
Configuration Settings
======================
Centralized configuration management using Pydantic Settings.
All hardcoded values are now configurable via environment variables.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    """Database connection settings."""
    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")
    
    HOST: str = Field(default="postgres", description="Database host")
    PORT: int = Field(default=5432, description="Database port")
    NAME: str = Field(default="hipaa", description="Database name")
    USER: str = Field(default="user", description="Database user")
    PASSWORD: str = Field(default="pass", description="Database password")
    MIN_POOL_SIZE: int = Field(default=2, description="Minimum connection pool size")
    MAX_POOL_SIZE: int = Field(default=10, description="Maximum connection pool size")
    MAX_RETRIES: int = Field(default=10, description="Max connection retries")
    RETRY_INTERVAL: int = Field(default=2, description="Seconds between retries")


class SearchSettings(BaseSettings):
    """Search and retrieval settings."""
    model_config = SettingsConfigDict(env_prefix="SEARCH_", extra="ignore")
    
    VECTOR_WEIGHT: float = Field(default=0.6, description="Weight for vector search in hybrid")
    KEYWORD_WEIGHT: float = Field(default=0.4, description="Weight for keyword search in hybrid")
    TOP_K: int = Field(default=15, description="Initial search results count")
    RERANK_TOP_K_CITATION: int = Field(default=2, description="Rerank top-k for citations")
    RERANK_TOP_K_EXPLANATION: int = Field(default=5, description="Rerank top-k for explanations")
    RERANK_TOP_K_REFERENCE: int = Field(default=7, description="Rerank top-k for reference lists")


class ChunkingSettings(BaseSettings):
    """ETL chunking settings."""
    model_config = SettingsConfigDict(env_prefix="CHUNK_", extra="ignore")
    
    MAX_CHUNK_SIZE: int = Field(default=7500, description="Maximum chunk size in characters")
    MIN_CHUNK_SIZE: int = Field(default=3000, description="Minimum chunk size for merging")
    OVERLAP_SIZE: int = Field(default=200, description="Overlap between chunks")


class ModelSettings(BaseSettings):
    """AI model settings."""
    model_config = SettingsConfigDict(extra="ignore")
    
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="Embedding model")
    EMBEDDING_DIMENSION: int = Field(default=1536, description="Embedding vector dimension")
    RERANK_MODEL: str = Field(default="gpt-4o-mini", description="Model for reranking")
    GENERATION_MODEL: str = Field(default="gpt-3.5-turbo", description="Model for answer generation")
    CLASSIFICATION_MODEL: str = Field(default="gpt-4o-mini", description="Model for query classification")
    MAX_TEXT_LENGTH: int = Field(default=8000, description="Max text length for embedding")


class Settings(BaseSettings):
    """Main settings aggregator."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Application settings
    APP_NAME: str = "HIPAA RAG API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # Data path for auto-loading
    DATA_JSON_PATH: str = Field(default="/app/data/hipaa_data.json", description="Path to chunks JSON")
    
    @property
    def database(self) -> DatabaseSettings:
        """Get database settings."""
        return DatabaseSettings()
    
    @property
    def search(self) -> SearchSettings:
        """Get search settings."""
        return SearchSettings()
    
    @property
    def chunking(self) -> ChunkingSettings:
        """Get chunking settings."""
        return ChunkingSettings()
    
    @property
    def models(self) -> ModelSettings:
        """Get model settings."""
        return ModelSettings()


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
