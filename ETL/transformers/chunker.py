"""
Chunker
=======
Intelligent chunking strategies for legal documents.
Handles large section splitting with subsection grouping and overlap.
"""

import re
from typing import Dict, Any, List, Optional

from config import get_settings, get_etl_logger

logger = get_etl_logger()
settings = get_settings()


class LegalChunker:
    """
    Intelligent chunker for legal documents.
    
    Implements strategies for splitting large sections while
    maintaining semantic coherence and legal structure.
    """
    
    def __init__(
        self,
        max_chunk_size: Optional[int] = None,
        min_chunk_size: Optional[int] = None,
        overlap_size: Optional[int] = None
    ):
        """
        Initialize the chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters for merging
            overlap_size: Characters to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size or settings.chunking.MAX_CHUNK_SIZE
        self.min_chunk_size = min_chunk_size or settings.chunking.MIN_CHUNK_SIZE
        self.overlap_size = overlap_size or settings.chunking.OVERLAP_SIZE
    
    def extract_references(self, text: str) -> List[str]:
        """
        Extract cross-references to other sections.
        
        Args:
            text: Text to search for references
        
        Returns:
            List of unique section references (e.g., ['164.530', '160.406'])
        """
        patterns = [
            r'§\s*(\d+\.\d+)',      # § 164.530
            r'§§\s*(\d+\.\d+)',     # §§ 160.406
            r'\$\\S\s*(\d+\.\d+)',  # $\S 164.530 (LaTeX)
        ]
        
        references = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            references.extend(matches)
        
        # Deduplicate while preserving order
        seen = set()
        unique_refs = []
        for ref in references:
            if ref not in seen:
                seen.add(ref)
                unique_refs.append(ref)
        
        return unique_refs
    
    def find_subsections(self, text: str) -> List[Dict[str, Any]]:
        """
        Find all lettered subsections in text.
        
        Args:
            text: Section text to parse
        
        Returns:
            List of subsection dicts with letter, title, text, etc.
        """
        # Pattern for main subsections: (a), (b), (c), etc.
        pattern = r'\n\(([a-z])\)\s+\*?([^*\n]+)\*?'
        matches = list(re.finditer(pattern, text))
        
        if len(matches) < 2:
            return []
        
        subsections = []
        for i, match in enumerate(matches):
            letter = match.group(1)
            title = match.group(2).strip()
            
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            subsection_text = text[start_pos:end_pos].strip()
            
            subsections.append({
                'letter': letter,
                'title': title,
                'text': subsection_text,
                'length': len(subsection_text),
                'position': i
            })
        
        return subsections
    
    def create_grouped_chunk(
        self,
        section_data: Dict[str, Any],
        subsections: List[Dict[str, Any]],
        group_index: int
    ) -> Dict[str, Any]:
        """
        Create a chunk from grouped subsections.
        
        Args:
            section_data: Parent section metadata
            subsections: List of subsections to combine
            group_index: Index for unique ID generation
        
        Returns:
            Chunk dictionary with combined content
        """
        # Combine texts
        combined_text = "\n\n".join(s['text'] for s in subsections)
        
        # Create letter range identifier
        letters = [s['letter'] for s in subsections]
        if len(letters) == 1:
            letter_range = f"({letters[0]})"
        else:
            letter_range = f"({letters[0]}-{letters[-1]})"
        
        # Create title
        if len(subsections) == 1:
            title_suffix = f"{letter_range} {subsections[0]['title']}"
        else:
            title_suffix = f"{letter_range} [{len(subsections)} subsections]"
        
        # Build chunk data
        chunk_data = section_data.copy()
        
        # Create unique ID
        chunk_id = f"{section_data['chunk_id']}_sub_g{group_index}_{letters[0]}"
        if len(letters) > 1:
            chunk_id += f"_{letters[-1]}"
        
        chunk_data.update({
            "chunk_id": chunk_id,
            "section_title": f"{section_data.get('section_title', '')} {title_suffix}",
            "text": combined_text,
            "references": self.extract_references(combined_text),
            "is_subchunk": True,
            "parent_section": section_data["chunk_id"],
            "subsection_marker": letter_range,
            "grouped_subsections": letters,
            "group_index": group_index,
        })
        
        return chunk_data
    
    def split_by_paragraphs(
        self,
        section_data: Dict[str, Any],
        text: str,
        subsection_suffix: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Split text by paragraphs with overlap.
        
        Fallback when no clear subsection structure exists.
        
        Args:
            section_data: Section metadata
            text: Text to split
            subsection_suffix: Suffix for chunk IDs
        
        Returns:
            List of chunk dictionaries
        """
        # Try paragraph splitting first
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Fallback to other split patterns if needed
        if len(paragraphs) <= 1:
            split_pattern = r'\n(?=\([A-Z]\)|\([ivxlcdm]+\)|\(\d+\))'
            segments = re.split(split_pattern, text)
            
            if len(segments) <= 1 or all(len(s) > self.max_chunk_size for s in segments):
                # Last resort: split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', text)
                paragraphs = sentences
            else:
                paragraphs = segments
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if adding paragraph exceeds limit
            if len(current_chunk) + len(para) + 2 > self.max_chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunks.append(self._create_paragraph_chunk(
                        section_data, current_chunk, chunk_index, subsection_suffix
                    ))
                    chunk_index += 1
                    
                    # Start new chunk with overlap
                    if len(current_chunk) > self.overlap_size:
                        overlap_text = current_chunk[-self.overlap_size:]
                        last_period = overlap_text.rfind('. ')
                        if last_period != -1:
                            overlap_text = overlap_text[last_period + 2:]
                        current_chunk = overlap_text + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    # Paragraph itself is too large - force split
                    chunks.extend(self._force_split_paragraph(
                        section_data, para, chunk_index, subsection_suffix
                    ))
                    chunk_index += len(chunks)
                    current_chunk = ""
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Save final chunk
        if current_chunk:
            chunks.append(self._create_paragraph_chunk(
                section_data, current_chunk, chunk_index, subsection_suffix
            ))
        
        return chunks
    
    def _create_paragraph_chunk(
        self,
        section_data: Dict[str, Any],
        text: str,
        index: int,
        subsection_suffix: str
    ) -> Dict[str, Any]:
        """Create a chunk from paragraph text."""
        suffix = f"{subsection_suffix}_p{index}" if subsection_suffix else f"_p{index}"
        
        chunk_data = section_data.copy()
        chunk_data.update({
            "chunk_id": f"{section_data['chunk_id']}{suffix}",
            "text": text.strip(),
            "references": self.extract_references(text),
            "is_subchunk": True,
            "parent_section": section_data["chunk_id"],
            "chunk_part": f"Part {index + 1}",
        })
        
        return chunk_data
    
    def _force_split_paragraph(
        self,
        section_data: Dict[str, Any],
        text: str,
        start_index: int,
        subsection_suffix: str
    ) -> List[Dict[str, Any]]:
        """Force split an oversized paragraph."""
        chunks = []
        chunk_index = start_index
        
        while text:
            if len(text) <= self.max_chunk_size:
                chunks.append(self._create_paragraph_chunk(
                    section_data, text, chunk_index, subsection_suffix
                ))
                break
            
            # Find good split point
            split_point = self.max_chunk_size
            last_period = text[:split_point].rfind('. ')
            if last_period != -1 and last_period > self.max_chunk_size * 0.5:
                split_point = last_period + 2
            
            chunk_piece = text[:split_point].strip()
            text = text[split_point:].strip()
            
            chunks.append(self._create_paragraph_chunk(
                section_data, chunk_piece, chunk_index, subsection_suffix
            ))
            chunk_index += 1
        
        return chunks
    
    def split_section(
        self,
        section_data: Dict[str, Any],
        text: str
    ) -> List[Dict[str, Any]]:
        """
        Intelligently split a large section into chunks.
        
        Strategy:
        1. Find all subsections (a), (b), (c), etc.
        2. Group subsections into chunks of ~5-7k chars
        3. If a group is still too large, split by paragraphs
        
        Args:
            section_data: Section metadata dictionary
            text: Full section text
        
        Returns:
            List of chunk dictionaries
        """
        # Find subsections
        subsections = self.find_subsections(text)
        
        # No clear subsections - fall back to paragraph splitting
        if len(subsections) < 2:
            return self.split_by_paragraphs(section_data, text)
        
        # Group subsections into chunks
        chunks = []
        current_group = []
        current_length = 0
        group_index = 0
        
        for subsec in subsections:
            # Would adding this subsection exceed the limit?
            if current_length + subsec['length'] > self.max_chunk_size:
                # Save current group
                if current_group:
                    chunks.append(self.create_grouped_chunk(
                        section_data, current_group, group_index
                    ))
                    group_index += 1
                    current_group = []
                    current_length = 0
                
                # Handle oversized single subsection
                if subsec['length'] > self.max_chunk_size:
                    logger.warning(
                        f"Subsection ({subsec['letter']}) too large "
                        f"({subsec['length']} chars) - splitting"
                    )
                    sub_chunks = self.split_by_paragraphs(
                        section_data,
                        subsec['text'],
                        subsection_suffix=f"({subsec['letter']})_g{group_index}"
                    )
                    chunks.extend(sub_chunks)
                    group_index += 1
                else:
                    # Start new group with this subsection
                    current_group.append(subsec)
                    current_length = subsec['length']
            else:
                # Add to current group
                current_group.append(subsec)
                current_length += subsec['length']
        
        # Save final group
        if current_group:
            chunks.append(self.create_grouped_chunk(
                section_data, current_group, group_index
            ))
        
        return chunks
    
    def needs_splitting(self, text: str) -> bool:
        """Check if text needs to be split."""
        return len(text) > self.max_chunk_size


# Convenience functions
def split_large_section(
    section_data: Dict[str, Any],
    text: str,
    **kwargs
) -> List[Dict[str, Any]]:
    """Split a large section using default chunker."""
    chunker = LegalChunker(**kwargs)
    return chunker.split_section(section_data, text)


def extract_references(text: str) -> List[str]:
    """Extract references using default chunker."""
    chunker = LegalChunker()
    return chunker.extract_references(text)
