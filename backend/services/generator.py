"""
Generator Service
=================
Generates answers for different query types using LLM.
"""

from typing import List, Dict, Optional

from openai import OpenAI

from config import get_settings, get_search_logger
from backend.models import QueryType

logger = get_search_logger()
settings = get_settings()


class GeneratorService:
    """
    Answer generation service.
    
    Generates appropriate responses based on query type:
    - Full text formatting
    - Citation extraction
    - Explanatory answers
    - Reference listings
    """
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        """
        Initialize the generator service.
        
        Args:
            openai_client: OpenAI client (creates default if None)
        """
        self.client = openai_client or OpenAI(
            api_key=settings.models.OPENAI_API_KEY
        )
        self.model = settings.models.GENERATION_MODEL
    
    async def generate_full_text_response(
        self,
        query: str,
        section_chunks: List[Dict]
    ) -> str:
        """
        Generate response for full text retrieval request.
        
        Args:
            query: User's query
            section_chunks: List of section chunks
        
        Returns:
            Formatted full section text
        """
        if not section_chunks:
            return "Section not found."
        
        main_chunk = section_chunks[0]
        section_id = main_chunk.get('section', 'Unknown')
        title = main_chunk.get('section_title', 'No title')
        
        # Build formatted text
        parts = [
            f"# {section_id}: {title}",
            ""
        ]
        
        for chunk in section_chunks:
            marker = chunk.get('subsection_marker', '')
            if marker:
                parts.append(f"## {marker}")
            parts.append(chunk.get('content', ''))
            parts.append("")
        
        return "\n".join(parts)
    
    async def generate_citation_response(
        self,
        query: str,
        relevant_sections: List[Dict]
    ) -> str:
        """
        Generate response with exact citations from regulations.
        
        Args:
            query: User's query
            relevant_sections: List of relevant section data
        
        Returns:
            Answer with direct citations
        """
        context = "\n\n".join([
            f"§ {sec['section']}:\n{sec['content'][:1500]}"
            for sec in relevant_sections
        ])
        
        prompt = f"""Based on these HIPAA regulation sections, provide a response with EXACT QUOTES from the text.

Context:
{context}

Question: {query}

Instructions:
- Use direct quotes with quotation marks
- Cite section numbers for each quote
- Be precise and legal-focused
- If the information isn't in the context, say so

Response:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    async def generate_explanation_response(
        self,
        query: str,
        relevant_sections: List[Dict]
    ) -> str:
        """
        Generate explanatory response with context.
        
        Args:
            query: User's query
            relevant_sections: List of relevant section data
        
        Returns:
            Clear explanatory answer
        """
        context = "\n\n".join([
            f"§ {sec['section']}:\n{sec['content'][:1200]}"
            for sec in relevant_sections
        ])
        
        prompt = f"""You are a HIPAA compliance expert. Answer the question based on these regulation sections.

Context:
{context}

Question: {query}

Instructions:
- Explain clearly in plain language
- Reference specific sections (§ XXX.XXX) when relevant
- Be accurate and helpful
- If unsure, say what you can determine from the context

Response:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    async def generate_reference_list_response(
        self,
        query: str,
        relevant_sections: List[Dict]
    ) -> str:
        """
        Generate response listing relevant sections.
        
        Args:
            query: User's query
            relevant_sections: List of relevant section data
        
        Returns:
            Formatted list of relevant sections
        """
        context = "\n\n".join([
            f"§ {sec['section']}:\n{sec['content'][:800]}"
            for sec in relevant_sections
        ])
        
        prompt = f"""Based on these HIPAA sections, create a reference list for the query.

Context:
{context}

Question: {query}

Instructions:
- List each relevant section with its number
- Provide a brief summary of what each section covers
- Organize logically (by relevance or topic)
- Include specific section citations

Response:"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        
        return response.choices[0].message.content
    
    async def generate(
        self,
        query: str,
        query_type: QueryType,
        sections: List[Dict]
    ) -> str:
        """
        Generate response based on query type.
        
        Args:
            query: User's query
            query_type: Type of query
            sections: Relevant sections
        
        Returns:
            Generated response
        """
        handlers = {
            QueryType.FULL_TEXT: self.generate_full_text_response,
            QueryType.CITATION: self.generate_citation_response,
            QueryType.EXPLANATION: self.generate_explanation_response,
            QueryType.REFERENCE_LIST: self.generate_reference_list_response,
        }
        
        handler = handlers.get(query_type, self.generate_explanation_response)
        return await handler(query, sections)


# Default service instance
_generator_service: Optional[GeneratorService] = None


def get_generator_service() -> GeneratorService:
    """Get or create generator service singleton."""
    global _generator_service
    if _generator_service is None:
        _generator_service = GeneratorService()
    return _generator_service
