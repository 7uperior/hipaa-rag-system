"""
Classifier Service
==================
Classifies user queries to route to appropriate handlers.
"""

from typing import Optional

from openai import OpenAI

from config import get_settings, get_search_logger
from backend.models import QueryType

logger = get_search_logger()
settings = get_settings()


class ClassifierService:
    """
    Query classification service.
    
    Uses LLM to classify queries into types:
    - FULL_TEXT: User wants complete section text
    - CITATION: User wants exact quotes
    - EXPLANATION: User wants understanding/clarification
    - REFERENCE_LIST: User wants list of relevant sections
    """
    
    CLASSIFICATION_PROMPT = """Classify this HIPAA question into ONE category:

Question: "{query}"

Categories:
- FULL_TEXT: User wants the complete/full/entire text of a specific section (keywords: "full text", "full contents", "complete text", "entire section", "all of section X", "show me 160.514", "give me everything from")
- CITATION: User wants exact quotes from regulations (keywords: "cite", "quote", "exact text", "what does section X say specifically")
- EXPLANATION: User wants understanding/clarification (keywords: "what is", "explain", "can I", "what does X mean", "how")
- REFERENCE_LIST: User wants list of relevant sections (keywords: "which sections", "list all", "what applies")

Respond with ONLY ONE WORD: full_text, citation, explanation, or reference_list"""
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        """
        Initialize the classifier service.
        
        Args:
            openai_client: OpenAI client (creates default if None)
        """
        self.client = openai_client or OpenAI(
            api_key=settings.models.OPENAI_API_KEY
        )
        self.model = settings.models.CLASSIFICATION_MODEL
    
    def classify(self, query: str) -> QueryType:
        """
        Classify a user query into a query type.
        
        Args:
            query: User's question
        
        Returns:
            QueryType enum value
        """
        prompt = self.CLASSIFICATION_PROMPT.format(query=query)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            
            classification = response.choices[0].message.content.strip().lower()
            
            # Map to QueryType
            type_map = {
                'full_text': QueryType.FULL_TEXT,
                'citation': QueryType.CITATION,
                'explanation': QueryType.EXPLANATION,
                'reference_list': QueryType.REFERENCE_LIST,
            }
            
            query_type = type_map.get(classification)
            
            if query_type is None:
                logger.warning(
                    f"Invalid classification '{classification}', "
                    "defaulting to explanation"
                )
                return QueryType.EXPLANATION
            
            logger.info(f"Query classified as: {query_type.value}")
            return query_type
        
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return QueryType.EXPLANATION


# Default service instance
_classifier_service: Optional[ClassifierService] = None


def get_classifier_service() -> ClassifierService:
    """Get or create classifier service singleton."""
    global _classifier_service
    if _classifier_service is None:
        _classifier_service = ClassifierService()
    return _classifier_service
