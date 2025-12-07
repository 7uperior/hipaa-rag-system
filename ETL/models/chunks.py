"""
ETL Data Models
===============
Pydantic models for HIPAA document chunks.
These models validate and structure the parsed legal content.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal


class BaseChunk(BaseModel):
    """Base model for all chunk types."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    part: str = Field(..., description="HIPAA Part number (160, 162, 164)")
    text: str = Field(default="", description="Main text content")
    
    class Config:
        extra = "forbid"


class SectionChunk(BaseChunk):
    """Model for regulation section chunks."""
    type: Literal["section"] = "section"
    
    # Hierarchy
    part_title: Optional[str] = Field(default=None, description="Part title")
    subpart: Optional[str] = Field(default=None, description="Subpart letter/number")
    subpart_title: Optional[str] = Field(default=None, description="Subpart title")
    section: str = Field(..., description="Section number with § prefix")
    section_title: Optional[str] = Field(default=None, description="Section title")
    
    # Subchunk metadata (for split sections)
    is_subchunk: bool = Field(default=False, description="Whether this is a subchunk of a larger section")
    parent_section: Optional[str] = Field(default=None, description="Parent section ID if subchunk")
    subsection_marker: Optional[str] = Field(default=None, description="Subsection marker like (a), (b)")
    chunk_part: Optional[str] = Field(default=None, description="Part X of Y for split chunks")
    grouped_subsections: Optional[List[str]] = Field(default=None, description="List of grouped subsection letters")
    group_index: Optional[int] = Field(default=None, description="Index for grouping order")
    
    # References
    references: List[str] = Field(default_factory=list, description="Cross-references to other sections")
    
    @field_validator("section")
    @classmethod
    def validate_section_format(cls, v: str) -> str:
        """Ensure section has § prefix."""
        if not v.startswith("§"):
            return f"§ {v}"
        return v


class PartMetadataChunk(BaseChunk):
    """Model for Part-level metadata (authority, source)."""
    type: Literal["part_metadata"] = "part_metadata"
    
    part_title: str = Field(..., description="Part title")
    authority: Optional[str] = Field(default=None, description="Legal authority citation")
    source: Optional[str] = Field(default=None, description="Federal Register source")


class SubpartMetadataChunk(BaseChunk):
    """Model for Subpart-level metadata."""
    type: Literal["subpart_metadata"] = "subpart_metadata"
    
    part_title: Optional[str] = Field(default=None, description="Part title")
    subpart: str = Field(..., description="Subpart identifier")
    subpart_title: Optional[str] = Field(default=None, description="Subpart title")
    source: Optional[str] = Field(default=None, description="Federal Register source")


class ReservedSectionChunk(BaseChunk):
    """Model for reserved (placeholder) sections."""
    type: Literal["reserved_section"] = "reserved_section"
    
    subpart: Optional[str] = Field(default=None, description="Subpart identifier")
    section: str = Field(..., description="Section number")
    text: str = Field(default="[Reserved]", description="Reserved text")


class ReservedSubpartChunk(BaseChunk):
    """Model for reserved (placeholder) subparts."""
    type: Literal["reserved_subpart"] = "reserved_subpart"
    
    subpart: str = Field(..., description="Subpart identifier")
    text: str = Field(default="[Reserved]", description="Reserved text")


# Type alias for any chunk type
ChunkType = SectionChunk | PartMetadataChunk | SubpartMetadataChunk | ReservedSectionChunk | ReservedSubpartChunk


def create_chunk(data: dict) -> ChunkType:
    """
    Factory function to create the appropriate chunk type.
    
    Args:
        data: Dictionary with chunk data including 'type' field
    
    Returns:
        Validated chunk model instance
    
    Raises:
        ValueError: If chunk type is unknown
    """
    chunk_type = data.get("type")
    
    type_map = {
        "section": SectionChunk,
        "part_metadata": PartMetadataChunk,
        "subpart_metadata": SubpartMetadataChunk,
        "reserved_section": ReservedSectionChunk,
        "reserved_subpart": ReservedSubpartChunk,
    }
    
    model_class = type_map.get(chunk_type)
    if model_class is None:
        raise ValueError(f"Unknown chunk type: {chunk_type}")
    
    return model_class(**data)
