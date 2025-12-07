"""
PostgreSQL Loader
=================
Loads parsed chunks into PostgreSQL with pgvector embeddings.
Handles schema creation, embedding generation, and batch insertion.
"""

import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path

import asyncpg
from openai import AsyncOpenAI
from pgvector.asyncpg import register_vector

from config import get_settings, get_etl_logger

logger = get_etl_logger()
settings = get_settings()


class PostgresLoader:
    """
    Loads HIPAA chunks into PostgreSQL with vector embeddings.
    
    Features:
    - Async operations for performance
    - Batch embedding generation
    - Schema management with pgvector
    - Helper functions for search
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the loader.
        
        Args:
            host: Database host (uses config default)
            database: Database name (uses config default)
            user: Database user (uses config default)
            password: Database password (uses config default)
            openai_api_key: OpenAI API key (uses config/env default)
        """
        self.host = host or settings.database.DB_HOST
        self.database = database or settings.database.DB_NAME
        self.user = user or settings.database.DB_USER
        self.password = password or settings.database.DB_PASSWORD
        
        api_key = openai_api_key or settings.models.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        self.openai_client = AsyncOpenAI(api_key=api_key)
        
        self.embedding_model = settings.models.EMBEDDING_MODEL
        self.embedding_dimension = settings.models.EMBEDDING_DIMENSION
        self.max_text_length = settings.models.MAX_TEXT_LENGTH
        
        self.conn: Optional[asyncpg.Connection] = None
    
    async def connect(self) -> asyncpg.Connection:
        """Establish database connection."""
        logger.info(f"🔌 Connecting to {self.host}/{self.database}...")
        
        self.conn = await asyncpg.connect(
            host=self.host,
            database=self.database,
            user=self.user,
            password=self.password
        )
        
        logger.info("✅ Connected!")
        return self.conn
    
    async def close(self):
        """Close database connection."""
        if self.conn:
            await self.conn.close()
            logger.info("🔒 Connection closed")
    
    async def create_schema(self):
        """Create database schema with pgvector extension."""
        logger.info("📦 Creating schema...")
        
        # Enable pgvector
        await self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await register_vector(self.conn)
        
        # Create main table
        await self.conn.execute("""
            DROP TABLE IF EXISTS hipaa_sections CASCADE;
            
            CREATE TABLE hipaa_sections (
                id SERIAL PRIMARY KEY,
                
                -- Unique chunk identifier
                chunk_id VARCHAR(100) UNIQUE NOT NULL,
                chunk_type VARCHAR(50) NOT NULL,
                
                -- Hierarchical structure
                part VARCHAR(10) NOT NULL,
                part_title TEXT,
                subpart VARCHAR(50),
                subpart_title TEXT,
                section VARCHAR(50),
                section_title TEXT,
                
                -- Subchunk metadata
                is_subchunk BOOLEAN DEFAULT FALSE,
                parent_section VARCHAR(100),
                subsection_marker VARCHAR(20),
                chunk_part VARCHAR(20),
                grouped_subsections TEXT[],
                group_index INTEGER,
                
                -- Content
                text TEXT NOT NULL,
                authority TEXT,
                source TEXT,
                cross_references TEXT[],
                
                -- Vector embedding
                embedding vector(1536),
                
                -- Timestamps
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Indexes for common queries
            CREATE INDEX idx_hipaa_part ON hipaa_sections(part);
            CREATE INDEX idx_hipaa_section ON hipaa_sections(section);
            CREATE INDEX idx_hipaa_chunk_type ON hipaa_sections(chunk_type);
            CREATE INDEX idx_hipaa_parent ON hipaa_sections(parent_section);
            CREATE INDEX idx_hipaa_fulltext ON hipaa_sections 
                USING gin(to_tsvector('english', text));
        """)
        
        logger.info("✅ Schema created!")
    
    async def create_vector_index(self):
        """Create IVFFlat index for vector similarity search."""
        logger.info("📊 Creating vector index...")
        
        await self.conn.execute("""
            CREATE INDEX IF NOT EXISTS hipaa_embedding_ivfflat_idx 
            ON hipaa_sections 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        logger.info("✅ Vector index created!")
    
    async def create_helper_functions(self):
        """Create PostgreSQL helper functions for search."""
        logger.info("📝 Creating helper functions...")
        
        # Citation formatter
        await self.conn.execute("""
            CREATE OR REPLACE FUNCTION format_citation(
                sec VARCHAR,
                marker VARCHAR,
                part_num VARCHAR
            ) RETURNS TEXT AS $$
            BEGIN
                IF marker IS NOT NULL AND marker != '' THEN
                    RETURN sec || ' ' || marker || ' (Part ' || part_num || ')';
                ELSE
                    RETURN sec || ' (Part ' || part_num || ')';
                END IF;
            END;
            $$ LANGUAGE plpgsql IMMUTABLE;
        """)
        
        # Full section context retriever
        await self.conn.execute("""
            CREATE OR REPLACE FUNCTION get_full_section_context(target_chunk_id VARCHAR)
            RETURNS TABLE (
                chunk_id VARCHAR,
                section VARCHAR,
                section_title TEXT,
                text TEXT,
                is_subchunk BOOLEAN,
                subsection_marker VARCHAR,
                group_index INTEGER
            ) AS $$
            DECLARE
                parent_id VARCHAR;
            BEGIN
                SELECT hs.parent_section INTO parent_id
                FROM hipaa_sections hs
                WHERE hs.chunk_id = target_chunk_id;
                
                IF parent_id IS NOT NULL THEN
                    RETURN QUERY
                    SELECT 
                        hs.chunk_id,
                        hs.section,
                        hs.section_title,
                        hs.text,
                        hs.is_subchunk,
                        hs.subsection_marker,
                        hs.group_index
                    FROM hipaa_sections hs
                    WHERE hs.chunk_id = parent_id
                       OR hs.parent_section = parent_id
                    ORDER BY hs.group_index NULLS FIRST, hs.chunk_id;
                ELSE
                    RETURN QUERY
                    SELECT 
                        hs.chunk_id,
                        hs.section,
                        hs.section_title,
                        hs.text,
                        hs.is_subchunk,
                        hs.subsection_marker,
                        hs.group_index
                    FROM hipaa_sections hs
                    WHERE hs.chunk_id = target_chunk_id
                       OR hs.parent_section = target_chunk_id
                    ORDER BY hs.group_index NULLS FIRST, hs.chunk_id;
                END IF;
            END;
            $$ LANGUAGE plpgsql;
        """)
        
        logger.info("✅ Helper functions created!")
    
    async def create_references_table(self):
        """Create cross-references table for GraphRAG."""
        logger.info("🔗 Creating references table...")
        
        await self.conn.execute("""
            DROP TABLE IF EXISTS section_references CASCADE;
            
            CREATE TABLE section_references (
                source_section VARCHAR(100) NOT NULL,
                target_section VARCHAR(100) NOT NULL,
                PRIMARY KEY (source_section, target_section)
            );
            
            CREATE INDEX idx_ref_source ON section_references(source_section);
            CREATE INDEX idx_ref_target ON section_references(target_section);
        """)
        
        logger.info("✅ References table created!")
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector or None on error
        """
        try:
            if len(text) > self.max_text_length:
                logger.warning(f"Text too long ({len(text)} chars), truncating")
                text = text[:self.max_text_length]
            
            response = await self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None
    
    def prepare_embedding_text(self, chunk: Dict[str, Any]) -> str:
        """
        Prepare text for embedding with hierarchical context.
        
        Args:
            chunk: Chunk dictionary
        
        Returns:
            Enriched text for embedding
        """
        chunk_type = chunk.get('type', 'section')
        parts = []
        
        if chunk_type == 'section':
            if chunk.get('part_title'):
                parts.append(f"Part {chunk['part']}: {chunk['part_title']}")
            
            if chunk.get('subpart_title'):
                parts.append(f"Subpart {chunk.get('subpart', '')}: {chunk['subpart_title']}")
            
            if chunk.get('section_title'):
                parts.append(f"{chunk.get('section', '')}: {chunk['section_title']}")
            
            if chunk.get('is_subchunk'):
                parent = chunk.get('parent_section')
                marker = chunk.get('subsection_marker')
                if parent and marker:
                    parts.append(f"Part of {parent}, subsection {marker}")
            
            parts.append(chunk.get('text', ''))
        
        elif chunk_type == 'part_metadata':
            parts.append(f"Part {chunk['part']}: {chunk.get('part_title', '')}")
            if chunk.get('authority'):
                parts.append(f"Legal Authority: {chunk['authority']}")
            if chunk.get('source'):
                parts.append(f"Source: {chunk['source']}")
            parts.append(chunk.get('text', ''))
        
        elif chunk_type == 'subpart_metadata':
            parts.append(f"Part {chunk['part']}")
            parts.append(f"Subpart {chunk.get('subpart', '')}: {chunk.get('subpart_title', '')}")
            if chunk.get('source'):
                parts.append(f"Source: {chunk['source']}")
            parts.append(chunk.get('text', ''))
        
        else:  # reserved
            parts.append(f"Part {chunk['part']}")
            parts.append(f"Reserved: {chunk.get('section', chunk.get('subpart', ''))}")
            parts.append(chunk.get('text', ''))
        
        return "\n".join(parts)
    
    async def load_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> Dict[str, int]:
        """
        Load chunks into database with embeddings.
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Number of chunks per batch
        
        Returns:
            Statistics dict with success/failure counts
        """
        total = len(chunks)
        successful = 0
        failed = 0
        
        logger.info(f"📤 Loading {total} chunks...")
        
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            insert_data = []
            
            for chunk in batch:
                # Generate embedding
                embedding_text = self.prepare_embedding_text(chunk)
                embedding = await self.get_embedding(embedding_text)
                
                if embedding is None:
                    failed += 1
                    continue
                
                # Prepare insert data
                insert_data.append((
                    chunk.get('chunk_id'),
                    chunk.get('type', 'section'),
                    chunk.get('part'),
                    chunk.get('part_title'),
                    chunk.get('subpart'),
                    chunk.get('subpart_title'),
                    chunk.get('section'),
                    chunk.get('section_title'),
                    chunk.get('is_subchunk', False),
                    chunk.get('parent_section'),
                    chunk.get('subsection_marker'),
                    chunk.get('chunk_part'),
                    chunk.get('grouped_subsections'),
                    chunk.get('group_index'),
                    chunk.get('text', ''),
                    chunk.get('authority'),
                    chunk.get('source'),
                    chunk.get('references', []),
                    embedding
                ))
            
            # Batch insert
            if insert_data:
                try:
                    await self.conn.executemany("""
                        INSERT INTO hipaa_sections (
                            chunk_id, chunk_type, part, part_title,
                            subpart, subpart_title, section, section_title,
                            is_subchunk, parent_section, subsection_marker,
                            chunk_part, grouped_subsections, group_index,
                            text, authority, source, cross_references, embedding
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                            $11, $12, $13, $14, $15, $16, $17, $18, $19
                        )
                    """, insert_data)
                    successful += len(insert_data)
                except Exception as e:
                    logger.error(f"Batch insert error: {e}")
                    failed += len(insert_data)
            
            logger.info(f"  Processed {min(i + batch_size, total)}/{total} chunks...")
        
        return {"successful": successful, "failed": failed}
    
    async def populate_references(self):
        """Populate cross-references table from chunk references."""
        logger.info("🔗 Populating cross-references...")
        
        await self.conn.execute("""
            INSERT INTO section_references (source_section, target_section)
            SELECT DISTINCT 
                chunk_id as source_section,
                UNNEST(cross_references) as target_section
            FROM hipaa_sections
            WHERE cross_references IS NOT NULL 
              AND array_length(cross_references, 1) > 0
            ON CONFLICT DO NOTHING;
        """)
        
        count = await self.conn.fetchval(
            "SELECT COUNT(*) FROM section_references"
        )
        logger.info(f"✅ Created {count} cross-references")


async def load_from_json(
    json_path: str,
    create_schema: bool = True
) -> Dict[str, Any]:
    """
    Load chunks from JSON file into database.
    
    Args:
        json_path: Path to JSON file with chunks
        create_schema: Whether to recreate schema
    
    Returns:
        Loading statistics
    """
    loader = PostgresLoader()
    
    try:
        await loader.connect()
        
        if create_schema:
            await loader.create_schema()
            await loader.create_references_table()
        
        # Load chunks
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        stats = await loader.load_chunks(chunks)
        
        # Create indexes and functions
        await loader.create_vector_index()
        await loader.create_helper_functions()
        await loader.populate_references()
        
        # Get final count
        count = await loader.conn.fetchval(
            "SELECT COUNT(*) FROM hipaa_sections"
        )
        stats['total_in_db'] = count
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ Loading complete!")
        logger.info(f"   Successful: {stats['successful']}")
        logger.info(f"   Failed: {stats['failed']}")
        logger.info(f"   Total in DB: {stats['total_in_db']}")
        logger.info(f"{'='*70}")
        
        return stats
    
    finally:
        await loader.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python postgres.py <chunks.json>")
        sys.exit(1)
    
    asyncio.run(load_from_json(sys.argv[1]))
