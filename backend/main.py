"""
HIPAA RAG API Server
====================

FastAPI application serving the HIPAA Expert RAG system.

Features:
- Hybrid search (vector + keyword)
- LLM-based reranking
- Query classification and routing
- Citation support
"""

import re
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends

from config import get_settings, setup_logging
from backend.db import db_pool, init_db, close_db, Queries
from backend.models import (
    Question,
    Answer,
    SearchResponse,
    SearchResult,
    SectionResponse,
    SectionChunk,
    HealthResponse,
    QueryType
)
from backend.dependencies import (
    get_search_service,
    get_reranker_service,
    get_classifier_service,
    get_generator_service
)
from backend.services import (
    SearchService,
    RerankerService,
    ClassifierService,
    GeneratorService
)

# Setup logging
logger = setup_logging(name="hipaa.api")
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting HIPAA RAG API...")
    await init_db()
    yield
    # Shutdown
    logger.info("👋 Shutting down...")
    await close_db()


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)


def extract_section_number(query: str) -> Optional[str]:
    """
    Extract section number from query.
    
    Args:
        query: User's question
    
    Returns:
        Section number without § prefix, or None
    """
    patterns = [
        r'§\s*(\d{3}\.\d{3,4})',       # § 160.514
        r'section\s+(\d{3}\.\d{3,4})',  # section 160.514
        r'\b(\d{3}\.\d{3,4})\b',        # 160.514
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


@app.post("/ask", response_model=Answer)
async def ask_question(
    q: Question,
    search_service: SearchService = Depends(get_search_service),
    reranker_service: RerankerService = Depends(get_reranker_service),
    classifier_service: ClassifierService = Depends(get_classifier_service),
    generator_service: GeneratorService = Depends(get_generator_service)
):
    """
    Answer a question about HIPAA regulations.
    
    Routes queries to appropriate handlers based on classification.
    """
    try:
        logger.info(f"📝 Question: {q.text[:100]}...")
        
        # Step 1: Classify query type
        query_type = classifier_service.classify(q.text)
        
        # Step 2: Handle full text requests specially
        if query_type == QueryType.FULL_TEXT:
            section_number = extract_section_number(q.text)
            
            if not section_number:
                return Answer(
                    answer="I couldn't identify a section number. Please specify like '160.514' or '§ 164.512'.",
                    sources=[],
                    query_type=query_type
                )
            
            logger.info(f"Retrieving full text for section: {section_number}")
            section_chunks = await search_service.get_full_section(section_number)
            
            if not section_chunks:
                return Answer(
                    answer=f"Section § {section_number} not found in database.",
                    sources=[],
                    query_type=query_type
                )
            
            answer = await generator_service.generate_full_text_response(
                q.text, section_chunks
            )
            sources = [section_chunks[0]['section']]
            
            return Answer(answer=answer, sources=sources, query_type=query_type)
        
        # Step 3: For other types - perform hybrid search
        candidates = await search_service.hybrid_search(
            q.text,
            top_k=settings.search.TOP_K
        )
        
        # Step 4: Rerank results based on query type
        rerank_top_k = {
            QueryType.CITATION: settings.search.RERANK_TOP_K_CITATION,
            QueryType.EXPLANATION: settings.search.RERANK_TOP_K_EXPLANATION,
            QueryType.REFERENCE_LIST: settings.search.RERANK_TOP_K_REFERENCE,
        }.get(query_type, settings.search.RERANK_TOP_K_EXPLANATION)
        
        relevant = reranker_service.rerank(q.text, candidates, top_k=rerank_top_k)
        
        if not relevant:
            return Answer(
                answer="No relevant information found in HIPAA documentation.",
                sources=[],
                query_type=query_type
            )
        
        # Step 5: Generate response
        answer = await generator_service.generate(q.text, query_type, relevant)
        
        # Collect sources
        sources = [chunk['section'] for chunk in relevant]
        
        return Answer(answer=answer, sources=sources, query_type=query_type)
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search/{query}", response_model=SearchResponse)
async def search(
    query: str,
    search_service: SearchService = Depends(get_search_service)
):
    """
    Debug endpoint for testing search functionality.
    """
    results = await search_service.hybrid_search(query, top_k=10)
    
    return SearchResponse(
        query=query,
        found=len(results),
        results=[
            SearchResult(
                section=r['section'],
                content=r['content'][:200],
                vector_score=r.get('vector_score', 0),
                keyword_score=r.get('keyword_score', 0),
                final_score=r.get('final_score', 0)
            )
            for r in results
        ]
    )


@app.get("/section/{section_id}", response_model=SectionResponse)
async def get_section(
    section_id: str,
    search_service: SearchService = Depends(get_search_service)
):
    """
    Retrieve complete section text with all subchunks.
    """
    section_chunks = await search_service.get_full_section(section_id)
    
    if not section_chunks:
        raise HTTPException(
            status_code=404,
            detail=f"Section {section_id} not found"
        )
    
    main_chunk = section_chunks[0]
    
    return SectionResponse(
        section=section_id,
        title=main_chunk.get('section_title'),
        part=main_chunk.get('part', ''),
        subpart=main_chunk.get('subpart'),
        chunks=[
            SectionChunk(
                chunk_id=chunk['chunk_id'],
                is_subchunk=chunk['is_subchunk'],
                subsection_marker=chunk.get('subsection_marker'),
                grouped_subsections=chunk.get('grouped_subsections'),
                text=chunk['content']
            )
            for chunk in section_chunks
        ],
        total_chunks=len(section_chunks)
    )


@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint with system status.
    """
    try:
        count = await db_pool.fetchval(Queries.COUNT_SECTIONS)
        
        return HealthResponse(
            status="running",
            database="postgresql_async",
            sections_loaded=count or 0,
            search_method="hybrid_vector_keyword_reranking",
            query_types=[t.value for t in QueryType]
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="running",
            database="postgresql_async",
            sections_loaded=0,
            search_method="hybrid_vector_keyword_reranking",
            query_types=[t.value for t in QueryType],
            database_error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
