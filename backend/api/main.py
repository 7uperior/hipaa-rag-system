"""FastAPI application."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

from backend.retrieval.retrieval import startup_db, shutdown_db, answer_question

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HIPAA RAG API",
    description="Question answering system for HIPAA regulations",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Request model for queries."""
    text: str


class QueryResponse(BaseModel):
    """Response model for queries."""
    answer: str
    sources: list[str]


@app.on_event("startup")
async def startup():
    """Initialize database connection pool."""
    await startup_db()
    logger.info("✅ API startup complete")


@app.on_event("shutdown")
async def shutdown():
    """Close database connection pool."""
    await shutdown_db()
    logger.info("🔒 API shutdown complete")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "status": "running",
        "message": "HIPAA RAG API",
        "version": "1.0.0",
        "endpoints": {
            "POST /ask": "Submit a question about HIPAA regulations"
        }
    }


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Answer HIPAA questions.
    
    Args:
        request: Query request with text field
        
    Returns:
        Answer and source citations
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    try:
        answer, sources = await answer_question(request.text)
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")