"""
HIPAA Vector Embedding Loader (Async)
=====================================

Description:
  This script manages the ingestion of parsed HIPAA data into a PostgreSQL 
  vector database using asyncpg for high performance.

Dependencies:
  - pip install asyncpg pgvector openai
  - PostgreSQL instance with 'pgvector' extension installed.
"""

import os
import json
import asyncio
import asyncpg
from openai import AsyncOpenAI
from pgvector.asyncpg import register_vector

# Initialize Async Client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def get_embedding(text):
    """Async wrapper to fetch embedding for a single text."""
    try:
        # Truncate to avoid token limits, similar to original script
        text = text[:6000]
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Error generating embedding: {e}")
        return None

async def main():
    print("🔌 Connecting to database...")
    try:
        conn = await asyncpg.connect(
            host="postgres",
            database="hipaa",
            user="user",
            password="pass"
        )
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    try:
        print("📦 Creating pgvector extension...")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Register the vector type so asyncpg handles arrays automatically
        await register_vector(conn)
        print("✅ Extension created and registered!")

        print("📦 Creating table...")
        await conn.execute("""
            DROP TABLE IF EXISTS hipaa_sections;
            
            CREATE TABLE hipaa_sections (
                id SERIAL PRIMARY KEY,
                section VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                embedding vector(1536)
            );
        """)
        print("✅ Table created!")

        print("📖 Loading hipaa_data_new.json...")
        with open('/app/hipaa_data.json', 'r') as f:
            sections = json.load(f)

        print(f"Found {len(sections)} sections")
        print("🧮 Generating embeddings and inserting into database (Async)...")
        
        batch_size = 10
        total_sections = len(sections)

        for i in range(0, total_sections, batch_size):
            batch = sections[i:i+batch_size]
            
            # 1. Create a list of async tasks for embeddings
            tasks = [get_embedding(item['content']) for item in batch]
            
            # 2. Run embedding generation in parallel for this batch
            embeddings = await asyncio.gather(*tasks)
            
            # 3. Prepare data for bulk insert (filter out failed embeddings)
            insert_data = []
            for item, emb in zip(batch, embeddings):
                if emb is not None:
                    insert_data.append((item['section'], item['content'], emb))

            # 4. Bulk insert using executemany
            if insert_data:
                await conn.executemany("""
                    INSERT INTO hipaa_sections (section, content, embedding)
                    VALUES ($1, $2, $3)
                """, insert_data)

            print(f"  Processed {min(i + batch_size, total_sections)}/{total_sections} sections...")

        print("✅ All sections loaded into PostgreSQL!")

        print("📊 Creating vector index for fast similarity search...")
        # Note: Index creation might take a moment
        await conn.execute("""
            CREATE INDEX ON hipaa_sections 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        print("✅ Index created!")

        count = await conn.fetchval("SELECT COUNT(*) FROM hipaa_sections;")
        print("\n📈 Database statistics:")
        print(f"   Total sections: {count}")

    finally:
        await conn.close()
        print("🔒 Connection closed")

if __name__ == "__main__":
    asyncio.run(main())