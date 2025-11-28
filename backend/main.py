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
            SELECT section, content, 
                   1 - (embedding <=> $1::vector) AS similarity
            FROM hipaa_sections
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
            'content': row['content'],
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


@app.post("/ask", response_model=Answer)
async def ask(q: Question) -> Answer:
    """
    Answer HIPAA-related questions using RAG approach.
    
    Supports two modes:
    - Citation mode: When query contains 'cite', 'quote', etc. - returns exact quotes
    - Standard mode: Provides comprehensive answers with section references
    
    Args:
        q: Question object containing query text
        
    Returns:
        Answer object with response text and source citations
    """
    try:
        # Detect if user is requesting specific citations
        needs_quotes = any(
            word in q.text.lower() 
            for word in ['cite', 'quote', 'exact', 'specific text', 'regulation text']
        )
        
        # Adjust search parameters based on query type
        if needs_quotes:
            candidates = await hybrid_search(q.text, top_k=15)
            relevant = rerank_results(q.text, candidates, top_k=7)
            context_length = 2000
        else:
            candidates = await hybrid_search(q.text, top_k=10)
            relevant = rerank_results(q.text, candidates, top_k=5)
            context_length = 1200
        
        if not relevant:
            return Answer(
                answer="No relevant information found in HIPAA documentation.",
                sources=[]
            )
        
        # Build context from relevant sections
        context_parts = [
            f"§ {chunk['section']}:\n{chunk['content'][:context_length]}"
            for chunk in relevant
        ]
        context = "\n\n---\n\n".join(context_parts)
        sources = [f"§ {chunk['section']}" for chunk in relevant]
        
        # Configure prompts based on query type
        if needs_quotes:
            system_prompt = """You are a HIPAA regulatory expert specializing in precise legal citations.

CRITICAL RULES for citation requests:
1. QUOTE exact regulation text verbatim - word for word
2. Use quotation marks "..." for all direct quotes
3. ALWAYS cite the specific section number (§ XXX.XXX) after each quote
4. Include multiple relevant quotes if the question asks for "texts" (plural)
5. Do NOT paraphrase when asked to cite - use exact wording from the regulations
6. If multiple sections are relevant, quote from all of them"""

            user_instruction = """IMPORTANT: The user is asking for SPECIFIC REGULATION TEXT.

Your response MUST include:
- Direct quotes from the regulations (in "quotation marks")
- Section numbers (§ XXX.XXX) cited after each quote
- Multiple quotes if multiple sections are relevant
- Exact wording from the provided context - do NOT paraphrase

Format each quote like this:
"[exact text from regulation]" (§ XXX.XXX)

Do NOT summarize or paraphrase. Quote the actual regulation text."""

            temperature = 0.0
            max_tokens = 1000
        else:
            system_prompt = """You are a HIPAA regulatory expert.

Rules:
1. Answer ONLY based on provided context
2. ALWAYS cite specific sections (§ XXX.XXX)
3. Be precise, factual, and comprehensive
4. If context doesn't contain answer, say so clearly"""

            user_instruction = """Instructions:
- Provide COMPREHENSIVE answer covering all relevant aspects
- Include specific details from regulations
- Cite multiple sections (§ XXX.XXX) throughout
- Be thorough and precise"""

            temperature = 0.3
            max_tokens = 800
        
        # Generate response using GPT
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Question: {q.text}

HIPAA Regulatory Context:
{context}

{user_instruction}"""}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return Answer(
            answer=response.choices[0].message.content,
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
            "search_method": "hybrid_vector_keyword_reranking"
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