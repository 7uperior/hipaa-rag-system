"""
Database Queries
================
SQL queries for HIPAA RAG operations.
Centralized query definitions for maintainability.
"""


class Queries:
    """SQL query definitions."""
    
    # === Vector Search ===
    VECTOR_SEARCH = """
        SELECT 
            section,
            text,
            1 - (embedding <=> $1::vector) AS similarity
        FROM hipaa_sections
        ORDER BY embedding <=> $1::vector
        LIMIT $2
    """
    
    # === Keyword Search ===
    KEYWORD_SEARCH_TEMPLATE = """
        SELECT 
            section, 
            text, 
            ({score_calc}) as score
        FROM hipaa_sections
        WHERE {where_clause}
        ORDER BY score DESC
        LIMIT $1
    """
    
    # === Full Section Retrieval ===
    GET_FULL_SECTION = """
        SELECT 
            chunk_id,
            section,
            section_title,
            text,
            is_subchunk,
            parent_section,
            subsection_marker,
            grouped_subsections,
            group_index,
            part,
            subpart
        FROM hipaa_sections
        WHERE 
            (chunk_id = $1 AND NOT is_subchunk)
            OR (parent_section = $1 AND is_subchunk)
        ORDER BY 
            CASE WHEN NOT is_subchunk THEN 0 ELSE 1 END,
            group_index NULLS LAST,
            chunk_id
    """
    
    # === Section Count ===
    COUNT_SECTIONS = """
        SELECT COUNT(*) FROM hipaa_sections
    """
    
    # === Cross References ===
    GET_RELATED_SECTIONS = """
        SELECT DISTINCT target_section
        FROM section_references
        WHERE source_section = ANY($1)
    """
    
    # === Full Text Search ===
    FULLTEXT_SEARCH = """
        SELECT 
            section,
            text,
            ts_rank(
                to_tsvector('english', text), 
                plainto_tsquery('english', $1)
            ) as rank
        FROM hipaa_sections
        WHERE to_tsvector('english', text) @@ plainto_tsquery('english', $1)
        ORDER BY rank DESC
        LIMIT $2
    """
    
    # === Hybrid Search (for reference) ===
    HYBRID_SEARCH = """
        WITH vector_results AS (
            SELECT 
                chunk_id, 
                section,
                text,
                1 - (embedding <=> $1::vector) as vector_score
            FROM hipaa_sections
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        ),
        keyword_results AS (
            SELECT 
                chunk_id,
                section,
                text,
                ts_rank(
                    to_tsvector('english', text), 
                    plainto_tsquery('english', $3)
                ) as keyword_score
            FROM hipaa_sections
            WHERE to_tsvector('english', text) @@ plainto_tsquery('english', $3)
        )
        SELECT 
            COALESCE(v.chunk_id, k.chunk_id) as chunk_id,
            COALESCE(v.section, k.section) as section,
            COALESCE(v.text, k.text) as text,
            COALESCE(v.vector_score, 0) * $4 + COALESCE(k.keyword_score, 0) * $5 as combined_score,
            v.vector_score,
            k.keyword_score
        FROM vector_results v
        FULL OUTER JOIN keyword_results k ON v.chunk_id = k.chunk_id
        ORDER BY combined_score DESC
        LIMIT $6
    """
    
    # === Section Validation ===
    VALIDATE_SECTIONS = """
        SELECT chunk_id 
        FROM hipaa_sections 
        WHERE chunk_id = ANY($1)
    """
    
    # === Get Section by ID ===
    GET_SECTION_BY_ID = """
        SELECT 
            chunk_id,
            section,
            section_title,
            text,
            part,
            subpart,
            is_subchunk,
            parent_section
        FROM hipaa_sections
        WHERE chunk_id = $1
    """


def build_keyword_search_query(keywords: list[str], conditions: list[str]) -> str:
    """
    Build dynamic keyword search query.
    
    Args:
        keywords: List of keywords for scoring
        conditions: WHERE clause conditions
    
    Returns:
        Complete SQL query string
    """
    # Build score calculation
    if keywords:
        score_parts = [
            f"(LENGTH(text) - LENGTH(REPLACE(LOWER(text), '{kw}', ''))) / LENGTH('{kw}')"
            for kw in keywords[:5]  # Limit to 5 keywords
        ]
        score_calc = ' + '.join(score_parts)
    else:
        score_calc = '0'
    
    # Build WHERE clause
    where_clause = ' OR '.join(conditions) if conditions else '1=1'
    
    return Queries.KEYWORD_SEARCH_TEMPLATE.format(
        score_calc=score_calc,
        where_clause=where_clause
    )
