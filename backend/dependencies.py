"""
Dependencies
============
FastAPI dependency injection providers.
"""

from functools import lru_cache
from typing import Generator

from openai import OpenAI

from config import Settings, get_settings
from backend.db import DatabasePool, db_pool
from backend.services import (
    SearchService,
    RerankerService,
    ClassifierService,
    GeneratorService
)


@lru_cache
def get_cached_settings() -> Settings:
    """Get cached settings instance."""
    return get_settings()


async def get_db() -> DatabasePool:
    """
    Dependency to get database pool.
    
    Ensures pool is connected before returning.
    """
    if not db_pool.is_connected:
        await db_pool.connect()
    return db_pool


def get_openai_client(
    settings: Settings = None
) -> OpenAI:
    """
    Dependency to get OpenAI client.
    
    Args:
        settings: Application settings
    
    Returns:
        Configured OpenAI client
    """
    settings = settings or get_settings()
    return OpenAI(api_key=settings.models.OPENAI_API_KEY)


# Service dependencies with proper instantiation
_search_service = None
_reranker_service = None
_classifier_service = None
_generator_service = None


def get_search_service() -> SearchService:
    """Get search service singleton."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service


def get_reranker_service() -> RerankerService:
    """Get reranker service singleton."""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service


def get_classifier_service() -> ClassifierService:
    """Get classifier service singleton."""
    global _classifier_service
    if _classifier_service is None:
        _classifier_service = ClassifierService()
    return _classifier_service


def get_generator_service() -> GeneratorService:
    """Get generator service singleton."""
    global _generator_service
    if _generator_service is None:
        _generator_service = GeneratorService()
    return _generator_service


def reset_services():
    """Reset all service singletons (useful for testing)."""
    global _search_service, _reranker_service, _classifier_service, _generator_service
    _search_service = None
    _reranker_service = None
    _classifier_service = None
    _generator_service = None
