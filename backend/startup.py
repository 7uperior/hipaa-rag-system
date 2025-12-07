#!/usr/bin/env python
"""
Backend Startup Script
======================
Initializes database with data and starts the API server.
"""

import asyncio
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, '/app')

import asyncpg
from openai import AsyncOpenAI

from config import get_settings, setup_logging

logger = setup_logging(name="hipaa.startup")
settings = get_settings()


async def wait_for_postgres(max_retries: int = 30, retry_interval: int = 2):
    """Wait for PostgreSQL to be ready."""
    logger.info("⏳ Waiting for PostgreSQL...")
    
    for attempt in range(max_retries):
        try:
            conn = await asyncpg.connect(
                host=settings.database.HOST,
                port=settings.database.PORT,
                database=settings.database.NAME,
                user=settings.database.USER,
                password=settings.database.PASSWORD
            )
            await conn.close()
            logger.info("✅ PostgreSQL is ready!")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                logger.info(f"   Attempt {attempt + 1}/{max_retries} - waiting...")
                await asyncio.sleep(retry_interval)
            else:
                logger.error(f"❌ PostgreSQL not ready after {max_retries} attempts: {e}")
                return False
    return False


async def check_data_loaded() -> bool:
    """Check if data is already in the database."""
    try:
        conn = await asyncpg.connect(
            host=settings.database.HOST,
            port=settings.database.PORT,
            database=settings.database.NAME,
            user=settings.database.USER,
            password=settings.database.PASSWORD
        )
        
        # Check if table exists and has data
        count = await conn.fetchval("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_name = 'hipaa_sections'
        """)
        
        if count == 0:
            await conn.close()
            return False
        
        row_count = await conn.fetchval("SELECT COUNT(*) FROM hipaa_sections")
        await conn.close()
        
        logger.info(f"📊 Found {row_count} sections in database")
        return row_count > 0
        
    except Exception as e:
        logger.warning(f"Could not check data: {e}")
        return False


async def create_schema(conn):
    """Create database schema."""
    logger.info("📦 Creating schema...")
    
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS hipaa_sections (
            id SERIAL PRIMARY KEY,
            chunk_id VARCHAR(100) UNIQUE NOT NULL,
            chunk_type VARCHAR(50) NOT NULL,
            part VARCHAR(10) NOT NULL,
            part_title TEXT,
            subpart VARCHAR(50),
            subpart_title TEXT,
            section VARCHAR(50),
            section_title TEXT,
            is_subchunk BOOLEAN DEFAULT FALSE,
            parent_section VARCHAR(100),
            subsection_marker VARCHAR(20),
            chunk_part VARCHAR(20),
            grouped_subsections TEXT[],
            group_index INTEGER,
            text TEXT NOT NULL,
            authority TEXT,
            source TEXT,
            cross_references TEXT[],
            embedding vector(1536),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    # Create indexes
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_hipaa_part ON hipaa_sections(part);")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_hipaa_section ON hipaa_sections(section);")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_hipaa_chunk_type ON hipaa_sections(chunk_type);")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_hipaa_parent ON hipaa_sections(parent_section);")
    await conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_hipaa_fulltext ON hipaa_sections 
        USING gin(to_tsvector('english', text));
    """)
    
    # Create references table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS section_references (
            source_section VARCHAR(100) NOT NULL,
            target_section VARCHAR(100) NOT NULL,
            PRIMARY KEY (source_section, target_section)
        );
    """)
    
    logger.info("✅ Schema created!")


async def get_embedding(client: AsyncOpenAI, text: str) -> list:
    """Generate embedding for text."""
    if len(text) > settings.models.MAX_TEXT_LENGTH:
        text = text[:settings.models.MAX_TEXT_LENGTH]
    
    response = await client.embeddings.create(
        model=settings.models.EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def prepare_embedding_text(chunk: dict) -> str:
    """Prepare text for embedding with context."""
    parts = []
    chunk_type = chunk.get('type', 'section')
    
    if chunk_type == 'section':
        if chunk.get('part_title'):
            parts.append(f"Part {chunk['part']}: {chunk['part_title']}")
        if chunk.get('subpart_title'):
            parts.append(f"Subpart {chunk.get('subpart', '')}: {chunk['subpart_title']}")
        if chunk.get('section_title'):
            parts.append(f"{chunk.get('section', '')}: {chunk['section_title']}")
        parts.append(chunk.get('text', ''))
    else:
        parts.append(f"Part {chunk['part']}")
        parts.append(chunk.get('text', ''))
    
    return "\n".join(parts)


async def load_data(json_path: str):
    """Load chunks from JSON into database."""
    logger.info(f"📤 Loading data from {json_path}...")
    
    if not os.path.exists(json_path):
        logger.error(f"❌ Data file not found: {json_path}")
        return False
    
    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    logger.info(f"   Found {len(chunks)} chunks to load")
    
    # Connect to database
    conn = await asyncpg.connect(
        host=settings.database.HOST,
        port=settings.database.PORT,
        database=settings.database.NAME,
        user=settings.database.USER,
        password=settings.database.PASSWORD
    )
    
    try:
        # Create schema
        await create_schema(conn)
        
        # Initialize OpenAI client
        client = AsyncOpenAI(api_key=settings.models.OPENAI_API_KEY)
        
        # Process chunks in batches
        batch_size = 5
        successful = 0
        failed = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            for chunk in batch:
                try:
                    # Generate embedding
                    embedding_text = prepare_embedding_text(chunk)
                    embedding = await get_embedding(client, embedding_text)
                    
                    # Convert embedding list to string format for pgvector
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    
                    # Insert into database
                    await conn.execute("""
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
                        ON CONFLICT (chunk_id) DO NOTHING
                    """,
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
                        embedding_str  # Pass string, not list
                    )
                    successful += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to insert chunk {chunk.get('chunk_id')}: {e}")
                    failed += 1
            
            logger.info(f"   Processed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks...")
        
        # Create vector index
        logger.info("📊 Creating vector index...")
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS hipaa_embedding_ivfflat_idx 
            ON hipaa_sections 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        # Populate references
        await conn.execute("""
            INSERT INTO section_references (source_section, target_section)
            SELECT DISTINCT 
                chunk_id as source_section,
                UNNEST(cross_references) as target_section
            FROM hipaa_sections
            WHERE cross_references IS NOT NULL 
              AND array_length(cross_references, 1) > 0
            ON CONFLICT DO NOTHING;
        """)
        
        logger.info(f"✅ Data loading complete!")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {failed}")
        
        return True
        
    finally:
        await conn.close()


async def init_database():
    """Initialize database with data."""
    logger.info("="*60)
    logger.info("🏥 HIPAA RAG Backend Startup")
    logger.info("="*60)
    
    # Wait for PostgreSQL
    if not await wait_for_postgres():
        logger.error("Cannot start without PostgreSQL")
        return False
    
    # Check if data needs to be loaded
    if not await check_data_loaded():
        logger.info("📂 Database empty, loading data...")
        
        data_path = settings.DATA_JSON_PATH
        
        # Also check common alternative paths
        if not os.path.exists(data_path):
            alternatives = [
                "/app/data/hipaa_data.json",
                "/app/data/chunks.json",
                "data/hipaa_data.json",
            ]
            for alt in alternatives:
                if os.path.exists(alt):
                    data_path = alt
                    break
        
        if os.path.exists(data_path):
            await load_data(data_path)
        else:
            logger.warning(f"⚠️ No data file found at {data_path}")
            logger.warning("   Database will be empty. Load data manually or provide hipaa_data.json")
    else:
        logger.info("✅ Data already loaded, skipping import")
    
    return True


def main():
    """Main startup routine."""
    # Run async initialization
    success = asyncio.run(init_database())
    
    if not success:
        sys.exit(1)
    
    logger.info("="*60)
    logger.info("🚀 Starting API server...")
    logger.info("="*60)
    
    # Start uvicorn (sync call - outside async context)
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
