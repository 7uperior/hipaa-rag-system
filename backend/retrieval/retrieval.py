"""Retrieval and reranking logic (async)."""

import os
import asyncpg
import re
import asyncio
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database connection pool
db_pool: asyncpg.Pool | None = None


async def startup_db():
    """Initialize database connection pool with retry logic."""
    global db_pool
    max_retries = 10
    retry_interval = 2

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
            return
        except (OSError, asyncpg.CannotConnectNowError) as e:
            if i < max_retries - 1:
                logger.warning(f"Database not ready, retrying in {retry_interval}s... ({i+1}/{max_retries})")
                await asyncio.sleep(retry_interval)
            else:
                logger.error(f"Failed to connect after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise


async def shutdown_db():
    """Close database connection pool."""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("🔒 Database connection pool closed")


async def vector_search(query: str, top_k: int = 10) -> list[dict]:
    """Semantic vector similarity search."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = response.data[0].embedding
    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT section, content, 
                   1 - (embedding <=> $1::vector) AS similarity
            FROM sections
            ORDER BY embedding <=> $1::vector
            LIMIT $2
        """, embedding_str, top_k)
    
    return [
        {
            'section': row['section'],
            'content': row['content'],
            'similarity': float(row['similarity']),
            'source': 'vector'
        }
        for row in rows
    ]


async def keyword_search(query: str, top_k: int = 10) -> list[dict]:
    """BM25-style keyword search with pattern matching."""
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
        keyword_conditions = ' OR '.join([f"content ILIKE '%{kw}%'" for kw in keywords[:5]])
        conditions.append(f"({keyword_conditions})")
    
    if not conditions:
        return []
    
    where_clause = ' OR '.join(conditions)
    score_parts = [
        f"(LENGTH(content) - LENGTH(REPLACE(LOWER(content), '{kw}', ''))) / LENGTH('{kw}')"
        for kw in keywords[:5]
    ]
    score_calc = ' + '.join(score_parts) if score_parts else '0'
    
    query_sql = f"""
        SELECT section, content, ({score_calc}) as score
        FROM sections
        WHERE {where_clause}
        ORDER BY score DESC
        LIMIT $1
    """
    
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(query_sql, top_k)
    
    return [
        {
            'section': row['section'],
            'content': row['content'],
            'similarity': float(row['score']) / 100,
            'source': 'keyword'
        }
        for row in rows
    ]


async def hybrid_search(query: str, top_k: int = 10) -> list[dict]:
    """Hybrid search: 60% vector + 40% keyword."""
    vector_results, keyword_results = await asyncio.gather(
        vector_search(query, top_k=top_k),
        keyword_search(query, top_k=top_k)
    )
    
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
    
    for section in combined:
        combined[section]['final_score'] = (
            0.6 * combined[section]['vector_score'] +
            0.4 * combined[section]['keyword_score']
        )
    
    results = sorted(combined.values(), key=lambda x: x['final_score'], reverse=True)
    return results[:top_k]


def rerank_results(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """LLM reranking with keyword boosting."""
    if len(candidates) <= top_k:
        return candidates[:top_k]
    
    query_lower = query.lower()
    boosted_sections = []
    
    if 'law enforcement' in query_lower:
        for candidate in candidates:
            if '164.512(f)' in candidate['section']:
                boosted_sections.append(candidate)
                break
    
    if 'business associate' in query_lower:
        for candidate in candidates:
            if '164.504' in candidate['section']:
                if candidate not in boosted_sections:
                    boosted_sections.append(candidate)
                break
    
    candidates_text = []
    for i, item in enumerate(candidates):
        section = item['section']
        content_preview = item['content'][:350]
        
        boost_marker = ""
        if item in boosted_sections:
            boost_marker = " ⭐ [HIGHLY RELEVANT]"
        elif '(' in section:
            boost_marker = " [SUBSECTION]"
        
        candidates_text.append(f"[{i}] § {section}{boost_marker}\n{content_preview}...")
    
    candidates_formatted = "\n\n".join(candidates_text)
    
    rerank_prompt = f"""You are a HIPAA expert. Rank these sections by relevance.

Query: "{query}"

RULES:
1. ⭐ sections MUST be ranked FIRST
2. Subsections (§ X.Y(a)) > parent sections (§ X.Y)
3. Exact keyword matches get priority

Sections:
{candidates_formatted}

Return JSON array of top {top_k} indices: [0, 3, 1, ...]

JSON only:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": rerank_prompt}],
            temperature=0.1
        )
        
        result_text = response.choices[0].message.content.strip()
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        ranked_indices = eval(result_text)
        
        reranked = [
            candidates[idx]
            for idx in ranked_indices[:top_k]
            if 0 <= idx < len(candidates)
        ]
        
        final_results = []
        for boosted in boosted_sections:
            if boosted not in final_results:
                final_results.append(boosted)
        
        for item in reranked:
            if item not in final_results and len(final_results) < top_k:
                final_results.append(item)
        
        return final_results[:top_k]
    
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        result = boosted_sections.copy()
        for c in candidates:
            if c not in result and len(result) < top_k:
                result.append(c)
        return result[:top_k]


async def answer_question(query: str) -> tuple[str, list[str]]:
    """Main RAG pipeline."""
    needs_quotes = any(
        word in query.lower() 
        for word in ['cite', 'quote', 'exact', 'specific text', 'regulation text']
    )
    
    if needs_quotes:
        candidates = await hybrid_search(query, top_k=15)
        relevant = rerank_results(query, candidates, top_k=7)
        context_length = 2000
    else:
        candidates = await hybrid_search(query, top_k=10)
        relevant = rerank_results(query, candidates, top_k=5)
        context_length = 1200
    
    if not relevant:
        return "No relevant information found.", []
    
    context_parts = [
        f"§ {chunk['section']}:\n{chunk['content'][:context_length]}"
        for chunk in relevant
    ]
    context = "\n\n---\n\n".join(context_parts)
    sources = [chunk['section'] for chunk in relevant]
    
    if needs_quotes:
        system_prompt = """HIPAA expert. Cite exact regulation text verbatim with quotation marks."""
        user_instruction = """Include direct quotes with section numbers: "[text]" (§ X.Y)"""
        temperature = 0.0
        max_tokens = 1000
    else:
        system_prompt = """HIPAA expert. Answer based on provided context. Always cite sections (§ X.Y)."""
        user_instruction = """Be comprehensive and precise. Cite all relevant sections."""
        temperature = 0.3
        max_tokens = 800
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Question: {query}

Context:
{context}

{user_instruction}"""}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    answer = response.choices[0].message.content
    return answer, sources