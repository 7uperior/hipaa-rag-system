"""
HIPAA RAG API Server
====================

Description:
  This is a FastAPI application that serves as the backend for the HIPAA 
  Expert RAG system. It implements a hybrid search algorithm (Vector + Keyword)
  and uses a two-step retrieval process (Search -> Rerank) to provide 
  accurate, cited answers to regulatory questions.

Key Features:
  - Hybrid Search: Combines pgvector cosine similarity with BM25-style keyword matching.
  - Reranking: Uses GPT-4o-mini to re-order search results for higher relevance.
  - Citations: Enforces strict sourcing of Section IDs (§) in answers.
  - Full Text Retrieval: Direct retrieval of complete section text with subchunks

Dependencies:
  - fastapi, uvicorn, asyncpg, openai, pydantic
  - Requires a running PostgreSQL instance with vector data loaded.

Usage:
  Run with uvicorn: `uvicorn main:app --reload`
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import os
import asyncpg
import re
from typing import List, Dict, Optional
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="HIPAA RAG API", version="1.0.0")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database connection pool
db_pool: Optional[asyncpg.Pool] = None


@app.on_event("startup")
async def startup():
    """Initialize database connection pool on startup with retry logic."""
    global db_pool
    max_retries = 10
    retry_interval = 2  # seconds

    for i in range(max_retries):
        try:
            db_pool = await asyncpg.create_pool(
                host="postgres",
                database="hipaa",
                user="user",
                password="pass",
                min_size=2,
                max_size=10
            )
            logger.info("✅ Database connection pool created")
            return  # Connection successful, exit function
        except (OSError, asyncpg.CannotConnectNowError) as e:
            if i < max_retries - 1:
                logger.warning(f"Database not ready yet, retrying in {retry_interval}s... ({i+1}/{max_retries})")
                await asyncio.sleep(retry_interval)
            else:
                logger.error(f"Failed to connect to database after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error creating database pool: {e}")
            raise


@app.on_event("shutdown")
async def shutdown():
    """Close database connection pool on shutdown."""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("🔒 Database connection pool closed")


class Question(BaseModel):
    """Request model for question endpoint."""
    text: str


class Answer(BaseModel):
    """Response model for question endpoint."""
    answer: str
    sources: List[str]


def extract_section_number(query: str) -> Optional[str]:
    """
    Извлекает номер секции из запроса БЕЗ символа §.
    
    Examples:
        "Give me full text of 160.514" -> "160.514"
        "Show me § 164.512" -> "164.512"
        "What is section 160.103 about" -> "160.103"
    
    Returns section ID without § prefix to match chunk_id format in database.
    """
    # Паттерны для поиска номеров секций
    patterns = [
        r'§\s*(\d{3}\.\d{3,4})',  # § 160.514
        r'section\s+(\d{3}\.\d{3,4})',  # section 160.514
        r'\b(\d{3}\.\d{3,4})\b',  # 160.514
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            # Return WITHOUT § prefix to match chunk_id format
            return match.group(1)
    
    return None


async def retrieve_full_section(section_id: str) -> List[Dict]:
    """
    Извлекает полный текст секции, включая все подчанки.
    
    Args:
        section_id: Номер секции БЕЗ § (например, "160.514" или "164.512")
    
    Returns:
        List of chunks ordered by group_index and chunk_id
    """
    async with db_pool.acquire() as conn:
        # Получаем основную секцию и все её подчанки
        # Используем chunk_id для поиска (без символа §)
        rows = await conn.fetch("""
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
        """, section_id)
        
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
                'grouped_subsections': list(row['grouped_subsections']) if row['grouped_subsections'] else None,
                'part': row['part'],
                'subpart': row['subpart']
            }
            for row in rows
        ]


async def vector_search(query: str, top_k: int = 10) -> List[Dict]:
    """
    Perform semantic vector similarity search using embeddings.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        
    Returns:
        List of matching sections with similarity scores
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding
    
    # Convert embedding list to pgvector format
    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT section, text, 
                   1 - (embedding <=> $1::vector) AS similarity
            FROM hipaa_sections
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """, embedding_str, top_k)
    
    return [
        {
            'section': row['section'],
            'content': row['text'],
            'similarity': float(row['similarity']),
            'source': 'vector'
        }
        for row in rows
    ]


async def keyword_search(query: str, top_k: int = 10) -> List[Dict]:
    """
    Perform BM25-style keyword search with pattern matching.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        
    Returns:
        List of matching sections with relevance scores
    """
    query_lower = query.lower()
    keywords = [k for k in re.findall(r'\b\w+\b', query_lower) if len(k) > 3]
    part_matches = re.findall(r'part\s+(\d+)', query_lower)
    section_matches = re.findall(r'§?\s*(\d+\.\d+)', query_lower)
    
    conditions = []
    
    if section_matches:
        section_pattern = '|'.join(section_matches)
        conditions.append(f"section ~ '{section_pattern}'")
    
    if part_matches:
        for part in part_matches:
            conditions.append(f"section LIKE '{part}%'")
    
    if keywords:
        keyword_conditions = ' OR '.join([f"text ILIKE '%{kw}%'" for kw in keywords[:5]])
        conditions.append(f"({keyword_conditions})")
    
    if not conditions:
        return []
    
    where_clause = ' OR '.join(conditions)
    score_parts = [
        f"(LENGTH(text) - LENGTH(REPLACE(LOWER(text), '{kw}', ''))) / LENGTH('{kw}')"
        for kw in keywords[:5]
    ]
    score_calc = ' + '.join(score_parts) if score_parts else '0'
    
    query_sql = f"""
        SELECT section, text, ({score_calc}) as score
        FROM hipaa_sections
        WHERE {where_clause}
        ORDER BY score DESC
        LIMIT $1
    """
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query_sql, top_k)
    
    return [
        {
            'section': row['section'],
            'content': row['text'],
            'similarity': float(row['score']) / 100,
            'source': 'keyword'
        }
        for row in rows
    ]


async def hybrid_search(query: str, top_k: int = 10) -> List[Dict]:
    """
    Perform hybrid search combining vector and keyword approaches.
    
    Combines semantic similarity (60% weight) with keyword matching (40% weight)
    for optimal retrieval performance.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        
    Returns:
        List of sections ranked by combined score
    """
    # Execute both searches in parallel
    vector_results, keyword_results = await asyncio.gather(
        vector_search(query, top_k=top_k),
        keyword_search(query, top_k=top_k)
    )
    
    # Combine results with weighted scoring
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
    
    # Calculate final score: 60% vector + 40% keyword
    for section in combined:
        combined[section]['final_score'] = (
            0.6 * combined[section]['vector_score'] +
            0.4 * combined[section]['keyword_score']
        )
    
    results = sorted(combined.values(), key=lambda x: x['final_score'], reverse=True)
    return results[:top_k]


def rerank_results(query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Rerank search results using LLM-based relevance scoring.
    
    Args:
        query: Original search query
        candidates: List of candidate sections from initial search
        top_k: Number of top results to return
        
    Returns:
        Reranked list of most relevant sections
    """
    if len(candidates) <= top_k:
        return candidates[:top_k]
    
    candidates_text = "\n".join([
        f"[{i}] § {item['section']}: {item['content'][:300]}..."
        for i, item in enumerate(candidates)
    ])
    
    rerank_prompt = f"""Given this user question about HIPAA regulations:
"{query}"

Rank the following {len(candidates)} section excerpts by relevance (most relevant first).

Sections:
{candidates_text}

Return ONLY a JSON array of the top {top_k} indices in order of relevance.
Example: [3, 0, 7, 2, 5]

Response (JSON array only):"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": rerank_prompt}],
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content.strip()
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        ranked_indices = eval(result_text)
        
        return [
            candidates[idx]
            for idx in ranked_indices[:top_k]
            if 0 <= idx < len(candidates)
        ]
    
    except Exception as e:
        logger.warning(f"Reranking failed: {e}, using original order")
        return candidates[:top_k]


async def classify_query_type(query: str) -> str:
    """
    Определяет тип запроса используя LLM.
    
    Возвращает: 'full_text' | 'citation' | 'explanation' | 'reference_list'
    """
    
    classification_prompt = f"""Classify this HIPAA question into ONE category:

Question: "{query}"

Categories:
- FULL_TEXT: User wants the complete/full/entire text of a specific section (keywords: "full text", "full contents", "complete text", "entire section", "all of section X", "show me 160.514", "give me everything from")
- CITATION: User wants exact quotes from regulations (keywords: "cite", "quote", "exact text", "what does section X say specifically")
- EXPLANATION: User wants understanding/clarification (keywords: "what is", "explain", "can I", "what does X mean", "how")
- REFERENCE_LIST: User wants list of relevant sections (keywords: "which sections", "list all", "what applies")

Respond with ONLY ONE WORD: full_text, citation, explanation, or reference_list"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": classification_prompt}],
            temperature=0.0,
            max_tokens=10
        )
        
        classification = response.choices[0].message.content.strip().lower()
        
        # Validate response
        valid_types = ['full_text', 'citation', 'explanation', 'reference_list']
        if classification not in valid_types:
            logger.warning(f"Invalid classification '{classification}', defaulting to explanation")
            return 'explanation'
            
        logger.info(f"Query classified as: {classification}")
        return classification
        
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return 'explanation'  # Default fallback


async def handle_full_text_request(query: str, section_chunks: List[Dict]) -> str:
    """
    РЕЖИМ 4: Полный текст секции
    
    Форматирует полный текст секции для пользователя.
    - Без LLM генерации
    - Прямой вывод текста из БД
    - Структурированное форматирование
    """
    if not section_chunks:
        return "❌ Section not found in database."
    
    # Основная информация
    main_chunk = section_chunks[0]
    section_id = main_chunk['section']
    section_title = main_chunk['section_title']
    
    # Формируем ответ
    parts = [
        f"# § {section_id}: {section_title}",
        f"\n**Part {main_chunk['part']}** | Subpart {main_chunk['subpart']}\n",
        "---\n"
    ]
    
    # Добавляем текст по частям
    for i, chunk in enumerate(section_chunks):
        if chunk['is_subchunk'] and chunk.get('subsection_marker'):
            # Подчанк с маркером
            if chunk.get('grouped_subsections'):
                markers = ', '.join(chunk['grouped_subsections'])
                parts.append(f"\n**Subsections ({markers}):**\n")
            else:
                parts.append(f"\n**Subsection ({chunk['subsection_marker']}):**\n")
        elif i > 0:
            parts.append("\n\n*[Continuation of section]*\n\n")
        
        parts.append(chunk['content'])
    
    # Добавляем информацию о количестве частей
    if len(section_chunks) > 1:
        parts.append(f"\n\n---\n*Total parts: {len(section_chunks)} (main section + {len(section_chunks)-1} subchunks)*")
    
    return "\n".join(parts)


async def handle_citation_request(query: str, relevant: list) -> str:
    """
    РЕЖИМ 1: Точные цитаты из регламента
    Температура: 0.0 (детерминированный)
    С проверкой цитат для защиты от галлюцинаций!
    """
    # Используем топ-2 чанка (компромисс между точностью и покрытием)
    context = "\n\n---\n\n".join([
        f"§ {chunk['section']}:\n{chunk['content'][:2000]}"
        for chunk in relevant[:2]
    ])
    
    system_prompt = """You are a HIPAA citation engine.

CRITICAL RULES:
1. Output EXACT verbatim text from the provided regulations
2. Use quotation marks "..." for all quotes
3. Cite the specific section number (§ XXX.XXX) after EVERY quote
4. Do NOT paraphrase or summarize
5. If exact text is not found, state "Exact text not found in provided context"

Format each citation like:
"[exact regulation text]" (§ XXX.XXX)"""

    user_prompt = f"""Question: {query}

HIPAA Regulatory Context:
{context}

Provide EXACT quotes with section citations."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        seed=42,
        max_tokens=1000
    )
    
    answer = response.choices[0].message.content
    
    # ⭐ ПРОВЕРКА ЦИТАТ (защита от галлюцинаций)
    if '"' in answer:
        import re
        quotes = re.findall(r'"([^"]+)"', answer)
        
        for quote in quotes:
            # Нормализуем пробелы и переносы строк
            normalized_quote = ' '.join(quote.split())
            
            # Проверяем в обоих чанках
            found = False
            for chunk in relevant[:2]:
                normalized_content = ' '.join(chunk['content'].split())
                if normalized_quote in normalized_content:
                    found = True
                    break
            
            if not found:
                logger.warning(f"⚠️ Quote verification FAILED: '{quote[:80]}...'")
                
                # Возвращаем безопасный ответ с доступными источниками
                available_sections = [f"§ {chunk['section']}" for chunk in relevant[:2]]
                return f"""⚠️ Citation Verification Failed

The system attempted to generate a quote but could not verify it in the source documents.

This question requires direct examination of the regulations.

**Available relevant sections:**
{chr(10).join(f"- {s}" for s in available_sections)}

**Suggestion:** Try asking for an explanation instead, or rephrase your question to be more specific about which section you need."""
    
    return answer


async def handle_explanation_request(query: str, relevant: list) -> str:
    """
    РЕЖИМ 2: Понятное объяснение
    Температура: 0.3 (сбалансированный)
    """
    context = "\n\n---\n\n".join([
        f"§ {chunk['section']}:\n{chunk['content'][:1200]}"
        for chunk in relevant[:5]  # Берем больше контекста
    ])
    
    system_prompt = """You are a HIPAA regulatory expert.

RULES:
1. Answer the question clearly and comprehensively
2. Synthesize information from multiple sections if needed
3. You MAY paraphrase for clarity
4. You MUST still cite sources using (§ XXX.XXX) format
5. Be accurate and factual - only use provided context"""

    user_prompt = f"""Question: {query}

HIPAA Regulatory Context:
{context}

Provide a clear, comprehensive answer with citations."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        seed=42,
        max_tokens=800
    )
    
    return response.choices[0].message.content


async def handle_reference_list_request(query: str, relevant: list) -> str:
    """
    РЕЖИМ 3: Список релевантных параграфов
    Температура: 0.1 (почти детерминированный)
    """
    context = "\n\n---\n\n".join([
        f"§ {chunk['section']}: {chunk['content'][:300]}..."
        for chunk in relevant[:7]  # Берем больше для полного списка
    ])
    
    system_prompt = """You are a HIPAA reference librarian.

RULES:
1. Do NOT answer the question in detail
2. Provide a STRUCTURED LIST of relevant section numbers
3. For each section, provide:
   - Section number (§ XXX.XXX)
   - One-sentence summary of what it covers
4. Order by relevance"""

    user_prompt = f"""Question: {query}

HIPAA Regulatory Context:
{context}

Provide a structured list of relevant sections."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        seed=42,
        max_tokens=600
    )
    
    return response.choices[0].message.content


@app.post("/ask", response_model=Answer)
async def ask(q: Question) -> Answer:
    """
    Answer HIPAA questions using intelligent routing.
    
    Routes queries to appropriate handlers:
    - Full Text: Complete text of specific sections
    - Citation: Exact quotes with section numbers
    - Explanation: Comprehensive synthesized answers
    - Reference List: Structured list of relevant sections
    """
    try:
        # Шаг 1: Классифицировать тип запроса
        query_type = await classify_query_type(q.text)
        logger.info(f"Query type: {query_type}")
        
        # Шаг 2: Обработка запроса на полный текст
        if query_type == 'full_text':
            section_number = extract_section_number(q.text)
            
            if not section_number:
                return Answer(
                    answer="❌ Could not identify section number. Please specify a section like '160.514' or '§ 164.512'.",
                    sources=[]
                )
            
            logger.info(f"Retrieving full text for section: {section_number}")
            section_chunks = await retrieve_full_section(section_number)
            
            if not section_chunks:
                return Answer(
                    answer=f"❌ Section § {section_number} not found in database. Please check the section number.",
                    sources=[]
                )
            
            answer = await handle_full_text_request(q.text, section_chunks)
            # Use section field from database (already has § prefix)
            sources = [section_chunks[0]['section']]
            
            return Answer(answer=answer, sources=sources)
        
        # Шаг 3: Для других типов - выполнить гибридный поиск
        candidates = await hybrid_search(q.text, top_k=15)
        
        # Шаг 4: Ре-ранжировать результаты
        if query_type == 'citation':
            relevant = rerank_results(q.text, candidates, top_k=2)
        elif query_type == 'explanation':
            relevant = rerank_results(q.text, candidates, top_k=5) 
        else:  # reference_list
            relevant = rerank_results(q.text, candidates, top_k=7)
        
        if not relevant:
            return Answer(
                answer="No relevant information found in HIPAA documentation.",
                sources=[]
            )
        
        # Шаг 5: Маршрутизировать к правильному обработчику
        if query_type == 'citation':
            answer = await handle_citation_request(q.text, relevant)
        elif query_type == 'explanation':
            answer = await handle_explanation_request(q.text, relevant)
        else:  # reference_list
            answer = await handle_reference_list_request(q.text, relevant)
        
        # Собрать источники (section field already contains § prefix)
        sources = [chunk['section'] for chunk in relevant]
        
        return Answer(
            answer=answer,
            sources=sources
        )
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Health check endpoint with system status."""
    try:
        async with db_pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM hipaa_sections;")
        
        return {
            "status": "running",
            "database": "postgresql_async",
            "sections_loaded": count,
            "search_method": "hybrid_vector_keyword_reranking",
            "query_types": ["full_text", "citation", "explanation", "reference_list"]
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "running",
            "database_error": str(e)
        }


@app.get("/search/{query}")
async def search(query: str):
    """
    Debug endpoint for testing search functionality.
    
    Args:
        query: Search query text
        
    Returns:
        Search results with scoring details
    """
    results = await hybrid_search(query, top_k=10)
    return {
        "query": query,
        "found": len(results),
        "results": [
            {
                "section": r['section'],
                "vector_score": r.get('vector_score', 0),
                "keyword_score": r.get('keyword_score', 0),
                "final_score": r.get('final_score', 0),
                "preview": r['content'][:200]
            }
            for r in results
        ]
    }


@app.get("/section/{section_id}")
async def get_section(section_id: str):
    """
    Direct endpoint for retrieving full section text.
    
    Args:
        section_id: Section number (e.g., "160.514")
        
    Returns:
        Complete section text with all subchunks
    """
    try:
        section_chunks = await retrieve_full_section(section_id)
        
        if not section_chunks:
            raise HTTPException(status_code=404, detail=f"Section {section_id} not found")
        
        # Format response
        main_chunk = section_chunks[0]
        
        return {
            "section": section_id,
            "title": main_chunk['section_title'],
            "part": main_chunk['part'],
            "subpart": main_chunk['subpart'],
            "chunks": [
                {
                    "chunk_id": chunk['chunk_id'],
                    "is_subchunk": chunk['is_subchunk'],
                    "subsection_marker": chunk['subsection_marker'],
                    "grouped_subsections": chunk['grouped_subsections'],
                    "text": chunk['content']
                }
                for chunk in section_chunks
            ],
            "total_chunks": len(section_chunks)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving section {section_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))