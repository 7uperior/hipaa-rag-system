"""
Chunking Tests
==============
Unit tests for the chunking strategy.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ETL.transformers.chunker import LegalChunker, extract_references


class TestLegalChunker:
    """Tests for LegalChunker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = LegalChunker(
            max_chunk_size=1000,
            min_chunk_size=300,
            overlap_size=50
        )
    
    def test_extract_references_basic(self):
        """Test basic reference extraction."""
        text = "See § 164.530 and § 160.406 for more information."
        refs = self.chunker.extract_references(text)
        
        assert "164.530" in refs
        assert "160.406" in refs
    
    def test_extract_references_duplicate(self):
        """Test that duplicates are removed."""
        text = "See § 164.530 and § 164.530 again."
        refs = self.chunker.extract_references(text)
        
        assert refs.count("164.530") == 1
    
    def test_extract_references_double_section(self):
        """Test §§ format."""
        text = "Per §§ 164.512, disclosure is permitted."
        refs = self.chunker.extract_references(text)
        
        assert "164.512" in refs
    
    def test_find_subsections(self):
        """Test subsection detection."""
        text = """
(a) *General rule.* A covered entity must comply.

(b) *Exception.* This does not apply when:
    (1) The individual requests it.
    (2) It is required by law.

(c) *Implementation.* The Secretary shall provide guidance.
"""
        subsections = self.chunker.find_subsections(text)
        
        assert len(subsections) == 3
        assert subsections[0]['letter'] == 'a'
        assert subsections[1]['letter'] == 'b'
        assert subsections[2]['letter'] == 'c'
    
    def test_needs_splitting_true(self):
        """Test needs_splitting returns True for large text."""
        large_text = "x" * 2000
        assert self.chunker.needs_splitting(large_text) is True
    
    def test_needs_splitting_false(self):
        """Test needs_splitting returns False for small text."""
        small_text = "x" * 500
        assert self.chunker.needs_splitting(small_text) is False
    
    def test_split_section_small(self):
        """Test that small sections are not split."""
        section_data = {
            "chunk_id": "164.530",
            "section": "§ 164.530",
            "section_title": "Administrative requirements",
        }
        text = "This is a small section that doesn't need splitting."
        
        chunks = self.chunker.split_section(section_data, text)
        
        # Should split by paragraphs as fallback (no subsections)
        assert len(chunks) >= 1
    
    def test_split_section_with_subsections(self):
        """Test splitting with subsection grouping."""
        section_data = {
            "chunk_id": "164.530",
            "section": "§ 164.530",
            "section_title": "Administrative requirements",
        }
        text = """
(a) *Standard: Personnel designations.* A covered entity must designate a privacy official.

(b) *Standard: Training.* A covered entity must train all workforce members.

(c) *Standard: Safeguards.* A covered entity must have appropriate safeguards.

(d) *Standard: Complaints.* A covered entity must provide a process for complaints.

(e) *Standard: Sanctions.* A covered entity must have sanctions for violations.

(f) *Standard: Mitigation.* A covered entity must mitigate harmful effects.

(g) *Standard: Retaliation.* A covered entity may not retaliate against individuals.
""" * 50  # Make it large enough to trigger splitting
        
        chunks = self.chunker.split_section(section_data, text)
        
        # Should create multiple chunks with grouped subsections
        assert len(chunks) > 1
        
        # Check that chunks have proper metadata
        for chunk in chunks:
            assert chunk.get('is_subchunk') is True
            assert chunk.get('parent_section') == "164.530"
    
    def test_grouped_chunk_id_format(self):
        """Test grouped chunk ID format."""
        section_data = {
            "chunk_id": "164.530",
            "section": "§ 164.530",
            "section_title": "Admin",
        }
        subsections = [
            {'letter': 'a', 'title': 'First', 'text': 'Content A', 'length': 100, 'position': 0},
            {'letter': 'b', 'title': 'Second', 'text': 'Content B', 'length': 100, 'position': 1},
        ]
        
        chunk = self.chunker.create_grouped_chunk(section_data, subsections, 0)
        
        assert 'a' in chunk['chunk_id']
        assert 'b' in chunk['chunk_id']
        assert chunk['grouped_subsections'] == ['a', 'b']


class TestExtractReferences:
    """Tests for standalone extract_references function."""
    
    def test_empty_text(self):
        """Test with empty text."""
        refs = extract_references("")
        assert refs == []
    
    def test_no_references(self):
        """Test text without references."""
        refs = extract_references("This text has no section references.")
        assert refs == []
    
    def test_multiple_references(self):
        """Test multiple different references."""
        text = """
        Under § 164.502(a), uses and disclosures are permitted.
        See also § 164.510 for uses for facility directory.
        Per § 160.103, definitions apply.
        """
        refs = extract_references(text)
        
        assert len(refs) == 3
        assert "164.502" in refs
        assert "164.510" in refs
        assert "160.103" in refs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
