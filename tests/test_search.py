"""
Search Tests
============
Unit tests for search functionality.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSearchService:
    """Tests for SearchService class."""
    
    @pytest.fixture
    def mock_openai(self):
        """Create mock OpenAI client."""
        mock = MagicMock()
        mock.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)]
        )
        return mock
    
    @pytest.fixture
    def search_service(self, mock_openai):
        """Create SearchService with mocked dependencies."""
        with patch('backend.services.search.OpenAI', return_value=mock_openai):
            from backend.services.search import SearchService
            service = SearchService(openai_client=mock_openai)
            return service
    
    def test_get_embedding(self, search_service):
        """Test embedding generation."""
        embedding = search_service.get_embedding("test query")
        
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_hybrid_search_combines_results(self, search_service):
        """Test that hybrid search combines vector and keyword results."""
        # Mock both search methods
        search_service.vector_search = AsyncMock(return_value=[
            {'section': '§ 164.502', 'content': 'Uses and disclosures', 'similarity': 0.9, 'source': 'vector'}
        ])
        search_service.keyword_search = AsyncMock(return_value=[
            {'section': '§ 164.502', 'content': 'Uses and disclosures', 'similarity': 0.7, 'source': 'keyword'},
            {'section': '§ 160.103', 'content': 'Definitions', 'similarity': 0.5, 'source': 'keyword'}
        ])
        
        results = await search_service.hybrid_search("uses and disclosures", top_k=5)
        
        # Should have combined results
        assert len(results) == 2
        
        # § 164.502 should be first (appears in both)
        assert results[0]['section'] == '§ 164.502'
        
        # Should have both scores
        assert results[0]['vector_score'] > 0
        assert results[0]['keyword_score'] > 0
    
    @pytest.mark.asyncio
    async def test_hybrid_search_weights(self, search_service):
        """Test that hybrid search applies correct weights."""
        search_service.vector_search = AsyncMock(return_value=[
            {'section': '§ 164.502', 'content': 'Content', 'similarity': 1.0, 'source': 'vector'}
        ])
        search_service.keyword_search = AsyncMock(return_value=[])
        
        results = await search_service.hybrid_search("test", top_k=5)
        
        # With 0.6 vector weight and 1.0 similarity
        expected_score = 0.6 * 1.0 + 0.4 * 0.0
        assert abs(results[0]['final_score'] - expected_score) < 0.01


class TestClassifierService:
    """Tests for ClassifierService class."""
    
    @pytest.fixture
    def mock_openai(self):
        """Create mock OpenAI client."""
        mock = MagicMock()
        return mock
    
    @pytest.fixture
    def classifier_service(self, mock_openai):
        """Create ClassifierService with mocked dependencies."""
        with patch('backend.services.classifier.OpenAI', return_value=mock_openai):
            from backend.services.classifier import ClassifierService
            service = ClassifierService(openai_client=mock_openai)
            return service
    
    def test_classify_full_text(self, classifier_service):
        """Test classification of full text request."""
        classifier_service.client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="full_text"))]
        )
        
        from backend.models import QueryType
        result = classifier_service.classify("Give me the full contents of 164.530")
        
        assert result == QueryType.FULL_TEXT
    
    def test_classify_explanation(self, classifier_service):
        """Test classification of explanation request."""
        classifier_service.client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="explanation"))]
        )
        
        from backend.models import QueryType
        result = classifier_service.classify("What does minimum necessary mean?")
        
        assert result == QueryType.EXPLANATION
    
    def test_classify_citation(self, classifier_service):
        """Test classification of citation request."""
        classifier_service.client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="citation"))]
        )
        
        from backend.models import QueryType
        result = classifier_service.classify("Quote the exact text about disclosures")
        
        assert result == QueryType.CITATION
    
    def test_classify_invalid_defaults_to_explanation(self, classifier_service):
        """Test that invalid classification defaults to explanation."""
        classifier_service.client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="invalid_type"))]
        )
        
        from backend.models import QueryType
        result = classifier_service.classify("Some query")
        
        assert result == QueryType.EXPLANATION


class TestRerankerService:
    """Tests for RerankerService class."""
    
    @pytest.fixture
    def mock_openai(self):
        """Create mock OpenAI client."""
        mock = MagicMock()
        return mock
    
    @pytest.fixture
    def reranker_service(self, mock_openai):
        """Create RerankerService with mocked dependencies."""
        with patch('backend.services.reranker.OpenAI', return_value=mock_openai):
            from backend.services.reranker import RerankerService
            service = RerankerService(openai_client=mock_openai)
            return service
    
    def test_rerank_returns_top_k(self, reranker_service):
        """Test that rerank returns correct number of results."""
        reranker_service.client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="[2, 0, 1]"))]
        )
        
        candidates = [
            {'section': '§ 164.502', 'content': 'A'},
            {'section': '§ 164.504', 'content': 'B'},
            {'section': '§ 164.506', 'content': 'C'},
        ]
        
        results = reranker_service.rerank("test", candidates, top_k=2)
        
        assert len(results) == 2
        assert results[0]['section'] == '§ 164.506'  # Index 2
        assert results[1]['section'] == '§ 164.502'  # Index 0
    
    def test_rerank_fewer_candidates_than_top_k(self, reranker_service):
        """Test rerank with fewer candidates than top_k."""
        candidates = [
            {'section': '§ 164.502', 'content': 'A'},
        ]
        
        results = reranker_service.rerank("test", candidates, top_k=5)
        
        # Should return all candidates without calling LLM
        assert len(results) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
