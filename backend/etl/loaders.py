"""Data loaders for JSON and PostgreSQL."""

import json
import asyncio
import os
import asyncpg
from openai import AsyncOpenAI


class JSONLoader:
    """Save chunks to JSON file."""
    
    def save(self, chunks: list[dict], output_path: str) -> None:
        """Save chunks to JSON.
        
        Args:
            chunks: List of section chunks
            output_path: Path to output JSON file
        """
        print(f"💾 Saving to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(chunks)} chunks\n")


class PostgreSQLLoader:
    """Load chunks into PostgreSQL with embeddings (async)."""
    
    def __init__(self):
        """Initialize loader."""
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pool = None
    
    async def connect(self) -> None:
        """Connect to PostgreSQL."""
        print("🔌 Connecting to database...")
        
        self.pool = await asyncpg.create_pool(
            host=os.getenv("POSTGRES_HOST", "postgres"),
            database=os.getenv("POSTGRES_DB", "hipaa"),
            user=os.getenv("POSTGRES_USER", "user"),
            password=os.getenv("POSTGRES_PASSWORD", "pass"),
            min_size=2,
            max_size=10
        )
        
        print("✅ Connected!")
    
    async def setup_database(self) -> None:
        """Create table and index."""
        print("📦 Creating pgvector extension...")
        
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        print("✅ Extension created!")
        
        print("📦 Creating table...")
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                DROP TABLE IF EXISTS sections;
                CREATE TABLE sections (
                    id SERIAL PRIMARY KEY,
                    section VARCHAR(100) NOT NULL,
                    parent VARCHAR(100),
                    content TEXT NOT NULL,
                    word_count INTEGER,
                    level INTEGER,
                    embedding vector(1536)
                );
            """)
        
        print("✅ Table created!")
    
    async def load(self, chunks: list[dict]) -> None:
        """Load chunks with embeddings.
        
        Args:
            chunks: List of section chunks
        """
        print(f"📖 Loading {len(chunks)} sections...")
        print("🧮 Generating embeddings and inserting (Async)...")
        
        tasks = []
        for i, chunk in enumerate(chunks):
            task = self._process_chunk(chunk, i + 1, len(chunks))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        print("✅ All sections loaded!")
        
        print("📊 Creating vector index...")
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS sections_embedding_idx 
                ON sections USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
        
        print("✅ Index created!")
    
    async def _process_chunk(self, chunk: dict, idx: int, total: int) -> None:
        """Process single chunk."""
        # Generate embedding
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk['content']
        )
        embedding = response.data[0].embedding
        
        # Convert to string format for pgvector
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
        
        # Insert into DB
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO sections 
                (section, parent, content, word_count, level, embedding)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, 
                chunk['section'],
                chunk['parent'],
                chunk['content'],
                chunk['word_count'],
                chunk['level'],
                embedding_str
            )
        
        if idx % 10 == 0:
            print(f"  Processed {idx}/{total} sections...")
    
    async def close(self) -> None:
        """Close connection."""
        if self.pool:
            await self.pool.close()
            print("🔒 Connection closed")