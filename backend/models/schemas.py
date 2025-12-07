"""
API Schemas
===========
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum


class QueryType(str, Enum):
    """Types of queries the system can handle."""
    FULL_TEXT = "full_text"
    CITATION = "citation"
    EXPLANATION = "explanation"
    REFERENCE_LIST = "reference_list"


class Question(BaseModel):
    """Request model for asking questions."""
    text: str = Field(
        ...,
        description="The question about HIPAA regulations",
        min_length=1,
        max_length=2000
    )


class Answer(BaseModel):
    """Response model for answered questions."""
    answer: str = Field(..., description="The generated answer")
    sources: List[str] = Field(
        default_factory=list,
        description="List of source section citations"
    )
    query_type: Optional[QueryType] = Field(
        default=None,
        description="Type of query that was processed"
    )


class SearchResult(BaseModel):
    """Model for a single search result."""
    section: str = Field(..., description="Section number")
    content: str = Field(..., description="Section text content")
    vector_score: float = Field(default=0.0, description="Vector similarity score")
    keyword_score: float = Field(default=0.0, description="Keyword match score")
    final_score: float = Field(default=0.0, description="Combined final score")
    source: Optional[str] = Field(default=None, description="Search source type")


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    query: str = Field(..., description="Original search query")
    found: int = Field(..., description="Number of results found")
    results: List[SearchResult] = Field(
        default_factory=list,
        description="List of search results"
    )


class SectionChunk(BaseModel):
    """Model for a section chunk in full section response."""
    chunk_id: str
    is_subchunk: bool
    subsection_marker: Optional[str] = None
    grouped_subsections: Optional[List[str]] = None
    text: str


class SectionResponse(BaseModel):
    """Response model for section retrieval endpoint."""
    section: str = Field(..., description="Section number")
    title: Optional[str] = Field(default=None, description="Section title")
    part: str = Field(..., description="Part number")
    subpart: Optional[str] = Field(default=None, description="Subpart letter")
    chunks: List[SectionChunk] = Field(
        default_factory=list,
        description="Section content chunks"
    )
    total_chunks: int = Field(..., description="Total number of chunks")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")
    database: str = Field(..., description="Database type")
    sections_loaded: int = Field(..., description="Number of sections in database")
    search_method: str = Field(..., description="Search method being used")
    query_types: List[str] = Field(
        default_factory=list,
        description="Supported query types"
    )
    database_error: Optional[str] = Field(
        default=None,
        description="Database error if any"
    )


class ErrorResponse(BaseModel):
    """Response model for error cases."""
    detail: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(default=None, description="Type of error")
