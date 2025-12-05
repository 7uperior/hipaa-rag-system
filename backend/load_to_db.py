"""
HIPAA Vector Embedding Loader (Async) - Legal RAG Optimized
Обновлено для поддержки группированных подчанков
"""

import os
import json
import asyncio
import asyncpg
from openai import AsyncOpenAI
from pgvector.asyncpg import register_vector
from typing import Optional, Dict, Any

# Initialize Async Client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === КОНСТАНТЫ ===
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
MAX_TEXT_LENGTH = 8000  # Уже не нужно урезать - все чанки < 8000
BATCH_SIZE = 10

# === УТИЛИТЫ ===

async def get_embedding(text: str) -> Optional[list]:
    """Генерация embedding с обработкой ошибок."""
    try:
        # Проверка на всякий случай
        if len(text) > MAX_TEXT_LENGTH:
            print(f"⚠️ Text too long ({len(text)} chars), truncating to {MAX_TEXT_LENGTH}")
            text = text[:MAX_TEXT_LENGTH]
        
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Error generating embedding: {e}")
        return None


def prepare_embedding_text(chunk: Dict[str, Any]) -> str:
    """
    Подготовка текста для embedding с учетом типа чанка.
    Добавляет иерархический контекст для лучшего поиска.
    
    ОБНОВЛЕНО: Учитывает подчанки с группировкой.
    """
    chunk_type = chunk['type']
    parts = []
    
    # 1. SECTION - самый важный тип
    if chunk_type == 'section':
        # Иерархия: Part -> Subpart -> Section
        if chunk.get('part_title'):
            parts.append(f"Part {chunk['part']}: {chunk['part_title']}")
        
        if chunk.get('subpart_title'):
            subpart = chunk.get('subpart', '')
            parts.append(f"Subpart {subpart}: {chunk['subpart_title']}")
        
        # Заголовок секции (может содержать информацию о подразделах)
        if chunk.get('section_title'):
            parts.append(f"{chunk['section']}: {chunk['section_title']}")
        
        # НОВОЕ: Если это подчанк, добавляем контекст
        if chunk.get('is_subchunk'):
            parent = chunk.get('parent_section')
            marker = chunk.get('subsection_marker')
            if parent and marker:
                parts.append(f"Part of {parent}, subsection {marker}")
        
        # Основной текст
        parts.append(chunk['text'])
    
    # 2. PART_METADATA - метаданные части
    elif chunk_type == 'part_metadata':
        parts.append(f"Part {chunk['part']}: {chunk['part_title']}")
        
        if chunk.get('authority'):
            parts.append(f"Legal Authority: {chunk['authority']}")
        
        if chunk.get('source'):
            parts.append(f"Source: {chunk['source']}")
        
        parts.append(chunk['text'])
    
    # 3. SUBPART_METADATA
    elif chunk_type == 'subpart_metadata':
        parts.append(f"Part {chunk['part']}")
        parts.append(f"Subpart {chunk['subpart']}: {chunk.get('subpart_title', '')}")
        
        if chunk.get('source'):
            parts.append(f"Source: {chunk['source']}")
        
        parts.append(chunk['text'])
    
    # 4 & 5. RESERVED (можно пропустить или сделать минимальный embedding)
    else:  # reserved_section или reserved_subpart
        parts.append(f"Part {chunk['part']}")
        parts.append(f"Reserved: {chunk.get('section', chunk.get('subpart', ''))}")
        parts.append(chunk['text'])
    
    return "\n".join(parts)


def safe_get(chunk: Dict, key: str, default=None):
    """Безопасное получение значения, возвращает None для пустых строк."""
    value = chunk.get(key, default)
    return value if value not in (None, '', []) else default


async def create_schema(conn: asyncpg.Connection):
    """Создание оптимизированной схемы БД с поддержкой подчанков."""
    
    print("📦 Creating pgvector extension...")
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    await register_vector(conn)
    
    print("📦 Creating table schema...")
    await conn.execute("""
        DROP TABLE IF EXISTS hipaa_sections CASCADE;
        
        CREATE TABLE hipaa_sections (
            -- Primary key
            id SERIAL PRIMARY KEY,
            
            -- Уникальный идентификатор чанка
            chunk_id VARCHAR(100) UNIQUE NOT NULL,
            
            -- Тип чанка (для фильтрации)
            chunk_type VARCHAR(50) NOT NULL,
            
            -- === ИЕРАРХИЧЕСКАЯ СТРУКТУРА ===
            part VARCHAR(10) NOT NULL,
            part_title TEXT,
            
            subpart VARCHAR(50),
            subpart_title TEXT,
            
            section VARCHAR(50),
            section_title TEXT,
            
            -- === НОВЫЕ ПОЛЯ ДЛЯ ПОДЧАНКОВ ===
            is_subchunk BOOLEAN DEFAULT FALSE,
            parent_section VARCHAR(100),
            subsection_marker VARCHAR(20),
            chunk_part VARCHAR(20),
            grouped_subsections TEXT[],
            group_index INTEGER,
            
            -- === КОНТЕНТ ===
            text TEXT NOT NULL,
            
            -- === МЕТАДАННЫЕ (для part_metadata) ===
            authority TEXT,
            source TEXT,
            
            -- === ПЕРЕКРЕСТНЫЕ ССЫЛКИ ===
            cross_references TEXT[],
            
            -- === ВЕКТОРНОЕ ПРЕДСТАВЛЕНИЕ ===
            embedding vector(1536),
            
            -- === СЛУЖЕБНАЯ ИНФОРМАЦИЯ ===
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- === CONSTRAINTS ===
            CONSTRAINT valid_chunk_type CHECK (
                chunk_type IN ('section', 'part_metadata', 'subpart_metadata', 
                              'reserved_section', 'reserved_subpart')
            )
        );
    """)
    
    print("📊 Creating indexes...")
    await conn.execute("""
        -- Индексы для иерархического поиска
        CREATE INDEX idx_part ON hipaa_sections(part);
        CREATE INDEX idx_subpart ON hipaa_sections(part, subpart);
        CREATE INDEX idx_section ON hipaa_sections(section) WHERE section IS NOT NULL;
        CREATE INDEX idx_chunk_type ON hipaa_sections(chunk_type);
        
        -- Индексы для работы с подчанками
        CREATE INDEX idx_is_subchunk ON hipaa_sections(is_subchunk);
        CREATE INDEX idx_parent_section ON hipaa_sections(parent_section) 
            WHERE parent_section IS NOT NULL;
        CREATE INDEX idx_subsection_marker ON hipaa_sections(subsection_marker)
            WHERE subsection_marker IS NOT NULL;
        CREATE INDEX idx_group_index ON hipaa_sections(group_index)
            WHERE group_index IS NOT NULL;
        
        -- Композитный индекс для частых запросов
        CREATE INDEX idx_part_type ON hipaa_sections(part, chunk_type);
        
        -- Полнотекстовый поиск (для гибридного RAG)
        CREATE INDEX idx_text_fts ON hipaa_sections 
            USING gin(to_tsvector('english', text));
        
        -- Поиск по перекрестным ссылкам
        CREATE INDEX idx_cross_references_gin ON hipaa_sections 
            USING gin(cross_references)
            WHERE cross_references IS NOT NULL AND array_length(cross_references, 1) > 0;
    """)
    
    print("✅ Schema created!")


async def load_chunks(conn: asyncpg.Connection, json_path: str, 
                     skip_reserved: bool = True):
    """
    Загрузка чанков в БД с генерацией embeddings.
    ОБНОВЛЕНО: Поддержка новых полей подчанков.
    
    Args:
        conn: Подключение к БД
        json_path: Путь к JSON файлу с чанками
        skip_reserved: Пропускать ли reserved чанки (экономит токены)
    """
    
    print(f"📖 Loading chunks from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Found {len(chunks)} chunks")
    
    # Статистика по подчанкам
    subchunks = [c for c in chunks if c.get('is_subchunk')]
    print(f"   • Regular chunks: {len(chunks) - len(subchunks)}")
    print(f"   • Subchunks: {len(subchunks)}")
    
    # Фильтрация reserved (опционально)
    if skip_reserved:
        original_count = len(chunks)
        chunks = [c for c in chunks if c['type'] not in ['reserved_section', 'reserved_subpart']]
        print(f"Filtered out {original_count - len(chunks)} reserved chunks")
    
    total_chunks = len(chunks)
    print(f"Will process {total_chunks} chunks")
    
    print("🧮 Generating embeddings and inserting into database...")
    
    successful_inserts = 0
    failed_inserts = 0
    
    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i:i+BATCH_SIZE]
        
        # 1. Подготовка текстов для embedding
        texts_for_embedding = [prepare_embedding_text(chunk) for chunk in batch]
        
        # 2. Параллельная генерация embeddings
        tasks = [get_embedding(text) for text in texts_for_embedding]
        embeddings = await asyncio.gather(*tasks)
        
        # 3. Подготовка данных для bulk insert
        insert_data = []
        for chunk, emb in zip(batch, embeddings):
            if emb is None:
                failed_inserts += 1
                continue
            
            # Подготовка данных с учетом НОВЫХ полей
            insert_data.append((
                chunk['chunk_id'],
                chunk['type'],
                chunk['part'],
                safe_get(chunk, 'part_title'),
                safe_get(chunk, 'subpart'),
                safe_get(chunk, 'subpart_title'),
                safe_get(chunk, 'section'),
                safe_get(chunk, 'section_title'),
                # НОВЫЕ ПОЛЯ
                chunk.get('is_subchunk', False),
                safe_get(chunk, 'parent_section'),
                safe_get(chunk, 'subsection_marker'),
                safe_get(chunk, 'chunk_part'),
                safe_get(chunk, 'grouped_subsections', []),
                chunk.get('group_index'),
                # ОСТАЛЬНОЕ
                chunk['text'],
                safe_get(chunk, 'authority'),
                safe_get(chunk, 'source'),
                safe_get(chunk, 'references', []),
                emb
            ))
        
        # 4. Bulk insert
        if insert_data:
            try:
                await conn.executemany("""
                    INSERT INTO hipaa_sections (
                        chunk_id, chunk_type, part, part_title, 
                        subpart, subpart_title, section, section_title,
                        is_subchunk, parent_section, subsection_marker, 
                        chunk_part, grouped_subsections, group_index,
                        text, authority, source, cross_references, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                """, insert_data)
                successful_inserts += len(insert_data)
            except Exception as e:
                print(f"❌ Error inserting batch: {e}")
                failed_inserts += len(insert_data)
        
        print(f"  ✅ Processed {min(i + BATCH_SIZE, total_chunks)}/{total_chunks} chunks...")
    
    print(f"\n✅ Chunk loading complete!")
    print(f"   Success: {successful_inserts}")
    print(f"   Failed: {failed_inserts}")


async def create_vector_index(conn: asyncpg.Connection):
    """Создание векторного индекса для быстрого similarity search."""
    
    print("📊 Creating vector index (this may take a minute)...")
    
    await conn.execute("""
        CREATE INDEX hipaa_embedding_ivfflat_idx ON hipaa_sections 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)
    
    print("✅ Vector index created!")


async def create_helper_functions(conn: asyncpg.Connection):
    """
    Создание вспомогательных функций для упрощенного поиска и цитирования.
    """
    
    print("🔧 Creating helper functions...")
    
    # Функция для получения полного контекста подчанка
    await conn.execute("""
        CREATE OR REPLACE FUNCTION get_full_section_context(input_chunk_id VARCHAR)
        RETURNS TABLE (
            chunk_id VARCHAR,
            section VARCHAR,
            section_title TEXT,
            subsection_marker VARCHAR,
            text TEXT,
            is_main_chunk BOOLEAN
        ) AS $$
        BEGIN
            -- Если это подчанк, находим все части родительской секции
            IF EXISTS (SELECT 1 FROM hipaa_sections WHERE chunk_id = input_chunk_id AND is_subchunk = TRUE) THEN
                RETURN QUERY
                SELECT 
                    h.chunk_id,
                    h.section,
                    h.section_title,
                    h.subsection_marker,
                    h.text,
                    h.chunk_id = input_chunk_id AS is_main_chunk
                FROM hipaa_sections h
                WHERE h.parent_section = (
                    SELECT parent_section FROM hipaa_sections WHERE chunk_id = input_chunk_id
                )
                OR h.chunk_id = (
                    SELECT parent_section FROM hipaa_sections WHERE chunk_id = input_chunk_id
                )
                ORDER BY h.chunk_id;
            ELSE
                -- Если это обычная секция, возвращаем только её
                RETURN QUERY
                SELECT 
                    h.chunk_id,
                    h.section,
                    h.section_title,
                    h.subsection_marker,
                    h.text,
                    TRUE AS is_main_chunk
                FROM hipaa_sections h
                WHERE h.chunk_id = input_chunk_id;
            END IF;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Функция для форматирования цитаты
    await conn.execute("""
        CREATE OR REPLACE FUNCTION format_citation(
            input_section VARCHAR,
            input_subsection_marker VARCHAR DEFAULT NULL,
            input_part VARCHAR DEFAULT NULL
        )
        RETURNS TEXT AS $$
        BEGIN
            IF input_subsection_marker IS NOT NULL THEN
                RETURN input_section || ' ' || input_subsection_marker;
            ELSE
                RETURN input_section;
            END IF;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    print("✅ Helper functions created!")


async def print_statistics(conn: asyncpg.Connection):
    """Вывод статистики по загруженным данным."""
    
    print("\n" + "="*70)
    print("📈 DATABASE STATISTICS")
    print("="*70)
    
    # Общее количество
    total = await conn.fetchval("SELECT COUNT(*) FROM hipaa_sections;")
    print(f"\n📊 Total chunks: {total}")
    
    # По типам
    print("\n📋 Chunks by type:")
    types = await conn.fetch("""
        SELECT chunk_type, COUNT(*) as cnt 
        FROM hipaa_sections 
        GROUP BY chunk_type
        ORDER BY cnt DESC;
    """)
    for row in types:
        print(f"   • {row['chunk_type']:<20} {row['cnt']:>5} chunks")
    
    # НОВОЕ: Статистика по подчанкам
    print("\n📦 Subchunk statistics:")
    subchunk_stats = await conn.fetch("""
        SELECT 
            is_subchunk,
            COUNT(*) as cnt,
            AVG(LENGTH(text))::INT as avg_length,
            MAX(LENGTH(text)) as max_length
        FROM hipaa_sections
        WHERE chunk_type = 'section'
        GROUP BY is_subchunk;
    """)
    for row in subchunk_stats:
        chunk_type = "Subchunks" if row['is_subchunk'] else "Regular chunks"
        print(f"   • {chunk_type:<20} {row['cnt']:>5} (avg: {row['avg_length']:>5} chars, max: {row['max_length']:>5})")
    
    # Группированные подчанки
    grouped_count = await conn.fetchval("""
        SELECT COUNT(*) FROM hipaa_sections 
        WHERE grouped_subsections IS NOT NULL AND array_length(grouped_subsections, 1) > 1;
    """)
    print(f"   • Grouped subchunks:  {grouped_count:>5} (multiple subsections combined)")
    
    # По частям
    print("\n📚 Chunks by Part:")
    parts = await conn.fetch("""
        SELECT part, part_title, COUNT(*) as cnt 
        FROM hipaa_sections 
        GROUP BY part, part_title
        ORDER BY part;
    """)
    for row in parts:
        title = row['part_title'] if row['part_title'] else 'N/A'
        if title and title != 'N/A':
            title = title[:50] + '...' if len(title) > 50 else title
        print(f"   • Part {row['part']:<4} {title:<55} {row['cnt']:>4} chunks")
    
    # Перекрестные ссылки
    refs_count = await conn.fetchval("""
        SELECT COUNT(*) FROM hipaa_sections 
        WHERE cross_references IS NOT NULL AND array_length(cross_references, 1) > 0;
    """)
    print(f"\n🔗 Chunks with cross-references: {refs_count}")
    
    # Самые связанные секции
    if refs_count > 0:
        print("\n🔗 Most referenced sections:")
        top_refs = await conn.fetch("""
            SELECT unnest(cross_references) as ref_section, COUNT(*) as ref_count
            FROM hipaa_sections
            WHERE cross_references IS NOT NULL
            GROUP BY ref_section
            ORDER BY ref_count DESC
            LIMIT 5;
        """)
        for row in top_refs:
            print(f"   • § {row['ref_section']:<15} referenced {row['ref_count']} times")
    
    print("\n" + "="*70)


async def print_example_queries(conn: asyncpg.Connection):
    """Примеры полезных запросов для работы с данными."""
    
    print("\n" + "="*70)
    print("💡 EXAMPLE QUERIES")
    print("="*70)
    
    print("\n1️⃣ Semantic Search (with vector similarity):")
    print("""
    SELECT 
        chunk_id,
        section,
        subsection_marker,
        section_title,
        LEFT(text, 100) || '...' as preview,
        1 - (embedding <=> $1::vector) as similarity
    FROM hipaa_sections
    WHERE chunk_type = 'section'
    ORDER BY embedding <=> $1::vector
    LIMIT 5;
    """)
    
    print("\n2️⃣ Get full context of a subchunk:")
    print("""
    SELECT * FROM get_full_section_context('164.512_sub_g0_c_i');
    """)
    
    print("\n3️⃣ Find all parts of a split section:")
    print("""
    SELECT 
        chunk_id,
        subsection_marker,
        grouped_subsections,
        group_index,
        LENGTH(text) as text_length
    FROM hipaa_sections
    WHERE parent_section = '164.512'
    ORDER BY group_index, chunk_id;
    """)
    
    print("\n4️⃣ Search with citation formatting:")
    print("""
    SELECT 
        format_citation(section, subsection_marker, part) as citation,
        section_title,
        LEFT(text, 150) || '...' as preview
    FROM hipaa_sections
    WHERE chunk_type = 'section'
      AND to_tsvector('english', text) @@ plainto_tsquery('english', 'patient authorization')
    LIMIT 5;
    """)
    
    print("\n5️⃣ Hybrid search (vector + keyword):")
    print("""
    WITH vector_results AS (
        SELECT chunk_id, 1 - (embedding <=> $1::vector) as vector_score
        FROM hipaa_sections
        ORDER BY embedding <=> $1::vector
        LIMIT 20
    ),
    keyword_results AS (
        SELECT chunk_id, ts_rank(to_tsvector('english', text), plainto_tsquery('english', $2)) as keyword_score
        FROM hipaa_sections
        WHERE to_tsvector('english', text) @@ plainto_tsquery('english', $2)
    )
    SELECT 
        h.chunk_id,
        h.section,
        h.subsection_marker,
        h.section_title,
        COALESCE(v.vector_score, 0) * 0.7 + COALESCE(k.keyword_score, 0) * 0.3 as combined_score
    FROM hipaa_sections h
    LEFT JOIN vector_results v ON h.chunk_id = v.chunk_id
    LEFT JOIN keyword_results k ON h.chunk_id = k.chunk_id
    WHERE v.chunk_id IS NOT NULL OR k.chunk_id IS NOT NULL
    ORDER BY combined_score DESC
    LIMIT 5;
    """)
    
    print("\n" + "="*70)


async def main():
    """Главная функция."""
    
    print("="*70)
    print("🏥 HIPAA Legal RAG - Database Loader (Updated for Grouped Chunks)")
    print("="*70)
    
    # === ПОДКЛЮЧЕНИЕ К БД ===
    print("\n🔌 Connecting to database...")
    try:
        conn = await asyncpg.connect(
            host=os.getenv("DB_HOST", "postgres"),
            database=os.getenv("DB_NAME", "hipaa"),
            user=os.getenv("DB_USER", "user"),
            password=os.getenv("DB_PASSWORD", "pass")
        )
        print("✅ Connected!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    try:
        # === СОЗДАНИЕ СХЕМЫ ===
        await create_schema(conn)
        
        # === ЗАГРУЗКА ДАННЫХ ===
        json_path = '/app/hipaa_chunks.json'
        if not os.path.exists(json_path):
            # Пробуем альтернативные пути
            for alt_path in [
                '/app/hipaa_chunks_grouped.json',
                '/app/hipaa_data.json', 
                'hipaa_chunks.json',
                'hipaa_chunks_grouped.json',
                'hipaa_data.json'
            ]:
                if os.path.exists(alt_path):
                    json_path = alt_path
                    break
        
        print(f"📂 Using file: {json_path}")
        await load_chunks(conn, json_path, skip_reserved=True)
        
        # === СОЗДАНИЕ ВЕКТОРНОГО ИНДЕКСА ===
        await create_vector_index(conn)
        
        # === СОЗДАНИЕ ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ ===
        await create_helper_functions(conn)
        
        # === СТАТИСТИКА ===
        await print_statistics(conn)
        
        # === ПРИМЕРЫ ЗАПРОСОВ ===
        await print_example_queries(conn)
        
    finally:
        await conn.close()
        print("\n🔒 Connection closed")
        print("="*70)


if __name__ == "__main__":
    asyncio.run(main())