"""
Search Service
==============
Implements vector, keyword, and hybrid search strategies.
"""

import re
import asyncio
from typing import List, Dict, Optional

from openai import OpenAI

from config import get_settings, get_search_logger
from backend.db import db_pool, Queries, build_keyword_search_query

logger = get_search_logger()
settings = get_settings()


class SearchService:
    """
    Search service for HIPAA regulations.
    
    Implements multiple search strategies:
    - Vector search using embeddings
    - Keyword search using pattern matching
    - Hybrid search combining both approaches
    """
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        """
        Initialize the search service.
        
        Args:
            openai_client: OpenAI client for embeddings (creates default if None)
        """
        self.client = openai_client or OpenAI(
            api_key=settings.models.OPENAI_API_KEY
        )
        self.embedding_model = settings.models.EMBEDDING_MODEL
        self.vector_weight = settings.search.VECTOR_WEIGHT
        self.keyword_weight = settings.search.KEYWORD_WEIGHT
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    async def vector_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Perform semantic vector similarity search.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of matching sections with similarity scores
        """
        # Generate query embedding
        query_embedding = self.get_embedding(query)
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Execute search
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(Queries.VECTOR_SEARCH, embedding_str, top_k)
        
        return [
            {
                'section': row['section'],
                'content': row['text'],
                'similarity': float(row['similarity']),
                'source': 'vector'
            }
            for row in rows
        ]
    
    async def keyword_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Perform BM25-style keyword search with pattern matching.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of matching sections with relevance scores
        """
        query_lower = query.lower()
        
        # Extract keywords (words > 3 chars)
        keywords = [k for k in re.findall(r'\b\w+\b', query_lower) if len(k) > 3]
        
        # Extract specific patterns
        part_matches = re.findall(r'part\s+(\d+)', query_lower)
        section_matches = re.findall(r'§?\s*(\d+\.\d+)', query_lower)
        
        conditions = []
        
        # Section number matching
        if section_matches:
            section_pattern = '|'.join(section_matches)
            conditions.append(f"section ~ '{section_pattern}'")
        
        # Part number matching
        if part_matches:
            for part in part_matches:
                conditions.append(f"section LIKE '{part}%'")
        
        # Keyword matching
        if keywords:
            keyword_conditions = ' OR '.join([
                f"text ILIKE '%{kw}%'" 
                for kw in keywords[:5]
            ])
            conditions.append(f"({keyword_conditions})")
        
        if not conditions:
            return []
        
        # Build and execute query
        query_sql = build_keyword_search_query(keywords, conditions)
        
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(query_sql, top_k)
        
        return [
            {
                'section': row['section'],
                'content': row['text'],
                'similarity': float(row['score']) / 100,  # Normalize
                'source': 'keyword'
            }
            for row in rows
        ]
    
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Perform hybrid search combining vector and keyword approaches.
        
        Combines semantic similarity (default 60% weight) with keyword
        matching (default 40% weight) for optimal retrieval.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of sections ranked by combined score
        """
        # Execute both searches in parallel
        vector_results, keyword_results = await asyncio.gather(
            self.vector_search(query, top_k=top_k),
            self.keyword_search(query, top_k=top_k)
        )
        
        # Combine results
        combined = {}
        
        for item in vector_results:
            section = item['section']
            combined[section] = {
                'section': section,
                'content': item['content'],
                'vector_score': item['similarity'],
                'keyword_score': 0.0
            }
        
        for item in keyword_results:
            section = item['section']
            if section in combined:
                combined[section]['keyword_score'] = item['similarity']
            else:
                combined[section] = {
                    'section': section,
                    'content': item['content'],
                    'vector_score': 0.0,
                    'keyword_score': item['similarity']
                }
        
        # Calculate final scores
        for section in combined:
            combined[section]['final_score'] = (
                self.vector_weight * combined[section]['vector_score'] +
                self.keyword_weight * combined[section]['keyword_score']
            )
        
        # Sort and return top results
        results = sorted(
            combined.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )
        
        return results[:top_k]
    
    async def get_full_section(
        self,
        section_id: str
    ) -> List[Dict]:
        """
        Retrieve full section text including all subchunks.
        
        Args:
            section_id: Section number without § prefix (e.g., "160.514")
        
        Returns:
            List of chunks ordered by group_index
        """
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(Queries.GET_FULL_SECTION, section_id)
        
        if not rows:
            return []
        
        return [
            {
                'chunk_id': row['chunk_id'],
                'section': row['section'],
                'section_title': row['section_title'],
                'content': row['text'],
                'is_subchunk': row['is_subchunk'],
                'subsection_marker': row['subsection_marker'],
                'grouped_subsections': (
                    list(row['grouped_subsections']) 
                    if row['grouped_subsections'] else None
                ),
                'part': row['part'],
                'subpart': row['subpart']
            }
            for row in rows
        ]
    
    async def get_related_sections(
        self,
        section_ids: List[str]
    ) -> List[str]:
        """
        Get sections related via cross-references.
        
        Args:
            section_ids: List of section IDs to find relations for
        
        Returns:
            List of related section IDs
        """
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(Queries.GET_RELATED_SECTIONS, section_ids)
        
        return [row['target_section'] for row in rows]
    
    async def validate_sections(
        self,
        section_ids: List[str]
    ) -> List[str]:
        """
        Validate that section IDs exist in the database.
        
        Args:
            section_ids: List of section IDs to validate
        
        Returns:
            List of valid section IDs that exist
        """
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(Queries.VALIDATE_SECTIONS, section_ids)
        
        return [row['chunk_id'] for row in rows]


# Default service instance
_search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """Get or create search service singleton."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service
