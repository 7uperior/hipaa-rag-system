"""
Reranker Service
================
LLM-based reranking for improved search relevance.
"""

from typing import List, Dict, Optional

from openai import OpenAI

from config import get_settings, get_search_logger

logger = get_search_logger()
settings = get_settings()


class RerankerService:
    """
    LLM-based reranking service.
    
    Uses GPT models to re-order search results by relevance
    to the original query.
    """
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        """
        Initialize the reranker service.
        
        Args:
            openai_client: OpenAI client (creates default if None)
        """
        self.client = openai_client or OpenAI(
            api_key=settings.models.OPENAI_API_KEY
        )
        self.model = settings.models.RERANK_MODEL
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
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
        
        # Format candidates for LLM
        candidates_text = "\n".join([
            f"[{i}] § {item['section']}: {item['content'][:300]}..."
            for i, item in enumerate(candidates)
        ])
        
        prompt = f"""Given this user question about HIPAA regulations:
"{query}"

Rank the following {len(candidates)} section excerpts by relevance (most relevant first).

Sections:
{candidates_text}

Return ONLY a JSON array of the top {top_k} indices in order of relevance.
Example: [3, 0, 7, 2, 5]

Response (JSON array only):"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            # Parse response
            result_text = response.choices[0].message.content.strip()
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            ranked_indices = eval(result_text)
            
            # Return reranked results
            return [
                candidates[idx]
                for idx in ranked_indices[:top_k]
                if 0 <= idx < len(candidates)
            ]
        
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, using original order")
            return candidates[:top_k]


# Default service instance
_reranker_service: Optional[RerankerService] = None


def get_reranker_service() -> RerankerService:
    """Get or create reranker service singleton."""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service
