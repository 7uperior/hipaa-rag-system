"""
Legal Document Parser
=====================
Parses linearized HIPAA text into structured chunks.
Handles Part, Subpart, Section detection and metadata extraction.
"""

import re
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from config import get_settings, get_etl_logger
from ETL.models import (
    SectionChunk,
    PartMetadataChunk,
    SubpartMetadataChunk,
    ReservedSectionChunk,
    ReservedSubpartChunk,
    ChunkType
)
from ETL.transformers.chunker import LegalChunker, extract_references

logger = get_etl_logger()
settings = get_settings()


class HIPAAParser:
    """
    Parser for HIPAA regulation text.
    
    Converts linearized PDF text into structured, validated chunks
    ready for embedding and storage.
    """
    
    def __init__(self, max_chunk_size: Optional[int] = None):
        """
        Initialize the parser.
        
        Args:
            max_chunk_size: Maximum chunk size (uses config default if None)
        """
        self.max_chunk_size = max_chunk_size or settings.chunking.MAX_CHUNK_SIZE
        self.chunker = LegalChunker(max_chunk_size=self.max_chunk_size)
        
        # State tracking
        self._reset_state()
    
    def _reset_state(self):
        """Reset parser state for new document."""
        self.current_part: Optional[str] = None
        self.current_part_title: Optional[str] = None
        self.current_subpart: Optional[str] = None
        self.current_subpart_title: Optional[str] = None
        
        self.current_section_data: Optional[Dict[str, Any]] = None
        self.section_text_buffer: List[str] = []
        self.section_start_line: Optional[int] = None
        
        self.awaiting_part_metadata = False
        self.awaiting_subpart_metadata = False
        self.part_metadata_buffer: List[str] = []
        self.subpart_metadata_buffer: List[str] = []
        self.part_metadata_start_line: Optional[int] = None
        self.subpart_metadata_start_line: Optional[int] = None
        
        self.pending_section_header: Optional[tuple] = None
        self.pending_section_line: Optional[int] = None
        
        self.all_chunks: List[ChunkType] = []
    
    def _save_section_chunk(self):
        """Save accumulated section buffer as chunk(s)."""
        if not self.current_section_data:
            return
        
        full_text = "\n".join(self.section_text_buffer).strip()
        
        if len(full_text) > self.max_chunk_size:
            logger.info(
                f"⚠️ Large section {self.current_section_data['chunk_id']}: "
                f"{len(full_text)} chars - splitting with grouping"
            )
            
            sub_chunks = self.chunker.split_section(
                self.current_section_data, full_text
            )
            
            logger.info(f"   ✓ Created {len(sub_chunks)} subchunks")
            
            for chunk_data in sub_chunks:
                try:
                    validated = SectionChunk(**chunk_data)
                    self.all_chunks.append(validated)
                except Exception as e:
                    logger.error(
                        f"Validation error for subchunk "
                        f"{chunk_data.get('chunk_id', 'UNKNOWN')}: {e}"
                    )
        else:
            # Normal section
            self.current_section_data["text"] = full_text
            self.current_section_data["references"] = extract_references(full_text)
            
            try:
                validated = SectionChunk(**self.current_section_data)
                self.all_chunks.append(validated)
            except Exception as e:
                logger.error(
                    f"Validation error for section "
                    f"{self.current_section_data.get('chunk_id', 'UNKNOWN')}: {e}"
                )
        
        # Clear buffer
        self.current_section_data = None
        self.section_text_buffer = []
        self.section_start_line = None
    
    def _save_part_metadata(self):
        """Save Part metadata chunk."""
        if not self.part_metadata_buffer or not self.current_part:
            self.part_metadata_buffer = []
            self.awaiting_part_metadata = False
            return
        
        full_text = "\n".join(self.part_metadata_buffer).strip()
        
        authority = None
        source = None
        
        for line in self.part_metadata_buffer:
            clean_line = re.sub(r'^\*{2}|\*{2}$', '', line)
            
            if re.match(r'^(AUTHORITY|Authority):\s*', clean_line):
                authority = re.sub(r'^(AUTHORITY|Authority):\s*', '', clean_line).strip()
            elif re.match(r'^(SOURCE|Source):\s*', clean_line):
                source = re.sub(r'^(SOURCE|Source):\s*', '', clean_line).strip()
        
        try:
            chunk = PartMetadataChunk(
                type="part_metadata",
                chunk_id=f"{self.current_part}_metadata",
                part=self.current_part,
                part_title=self.current_part_title or "",
                authority=authority,
                source=source,
                text=full_text
            )
            self.all_chunks.append(chunk)
        except Exception as e:
            logger.error(f"Validation error for Part metadata: {e}")
        
        self.part_metadata_buffer = []
        self.awaiting_part_metadata = False
        self.part_metadata_start_line = None
    
    def _save_subpart_metadata(self):
        """Save Subpart metadata chunk."""
        if not self.subpart_metadata_buffer or not self.current_subpart:
            self.subpart_metadata_buffer = []
            self.awaiting_subpart_metadata = False
            return
        
        full_text = "\n".join(self.subpart_metadata_buffer).strip()
        
        source = None
        for line in self.subpart_metadata_buffer:
            if re.match(r'^(SOURCE|Source):\s*', line):
                source = re.sub(r'^(SOURCE|Source):\s*', '', line).strip()
        
        try:
            chunk = SubpartMetadataChunk(
                type="subpart_metadata",
                chunk_id=f"{self.current_part}_{self.current_subpart}_metadata",
                part=self.current_part,
                subpart=self.current_subpart,
                subpart_title=self.current_subpart_title,
                source=source,
                text=full_text
            )
            self.all_chunks.append(chunk)
        except Exception as e:
            logger.error(f"Validation error for Subpart metadata: {e}")
        
        self.subpart_metadata_buffer = []
        self.awaiting_subpart_metadata = False
        self.subpart_metadata_start_line = None
    
    def _handle_pending_section(self, line: str, line_num: int) -> bool:
        """
        Handle multi-line section headers.
        
        Returns:
            True if line was consumed by pending header logic
        """
        if not self.pending_section_header:
            return False
        
        # Check if this starts a new structural element
        if (line.startswith("# PART") or 
            line.startswith("## Subpart") or 
            re.match(r'^##\s+[^\d]+\d', line)):
            
            # Finalize pending section with incomplete title
            sec_num, incomplete_title = self.pending_section_header
            self._create_section_data(sec_num, incomplete_title or "No Title", line_num)
            self.pending_section_header = None
            self.pending_section_line = None
            return False  # Let the line be processed normally
        
        # Append to title
        sec_num, incomplete_title = self.pending_section_header
        completed_title = incomplete_title + " " + line if incomplete_title else line
        
        # Check if title is complete (ends with period)
        if completed_title.endswith('.'):
            self._create_section_data(sec_num, completed_title, self.pending_section_line)
            self.pending_section_header = None
            self.pending_section_line = None
        else:
            self.pending_section_header = (sec_num, completed_title)
        
        return True
    
    def _create_section_data(self, sec_num: str, title: str, line_num: int):
        """Create section data dictionary."""
        self.current_section_data = {
            "type": "section",
            "part": self.current_part,
            "part_title": self.current_part_title,
            "subpart": self.current_subpart,
            "subpart_title": self.current_subpart_title,
            "section": f"§ {sec_num}",
            "section_title": title,
            "chunk_id": sec_num,
            "text": "",
            "references": []
        }
        self.section_start_line = line_num
    
    def parse_file(self, filepath: str) -> List[ChunkType]:
        """
        Parse a linearized text file into chunks.
        
        Args:
            filepath: Path to linearized text file
        
        Returns:
            List of validated chunk objects
        """
        self._reset_state()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        return self._parse_lines(lines)
    
    def parse_text(self, text: str) -> List[ChunkType]:
        """
        Parse text content into chunks.
        
        Args:
            text: Linearized document text
        
        Returns:
            List of validated chunk objects
        """
        self._reset_state()
        lines = text.split('\n')
        return self._parse_lines([line + '\n' for line in lines])
    
    def _parse_lines(self, lines: List[str]) -> List[ChunkType]:
        """Parse list of lines into chunks."""
        
        for line_num, line in enumerate(lines, start=1):
            line = line.strip()
            
            # Skip empty lines and separators
            if not line or line == "---":
                continue
            
            # Handle pending multi-line section headers
            if self._handle_pending_section(line, line_num):
                continue
            
            # === PART Detection ===
            part_match = re.match(r'^#\s*PART\s+(\d+)\s*[-–—]\s*(.+)$', line)
            if part_match:
                self._save_section_chunk()
                self._save_part_metadata()
                self._save_subpart_metadata()
                
                self.current_part = part_match.group(1)
                self.current_part_title = part_match.group(2).strip()
                self.current_subpart = None
                self.current_subpart_title = None
                
                self.awaiting_part_metadata = True
                self.part_metadata_start_line = line_num
                continue
            
            # === SUBPART Detection ===
            subpart_match = re.match(
                r'^##\s*Subpart\s+([A-Z])\s*[-–—]\s*(.+)$', 
                line, 
                re.IGNORECASE
            )
            if subpart_match:
                self._save_section_chunk()
                self._save_part_metadata()
                self._save_subpart_metadata()
                
                self.current_subpart = subpart_match.group(1)
                self.current_subpart_title = subpart_match.group(2).strip()
                
                # Check for reserved subpart
                if "[Reserved]" in line:
                    try:
                        chunk = ReservedSubpartChunk(
                            type="reserved_subpart",
                            chunk_id=f"{self.current_part}_{self.current_subpart}_reserved",
                            part=self.current_part,
                            subpart=self.current_subpart,
                            text="[Reserved]"
                        )
                        self.all_chunks.append(chunk)
                    except Exception as e:
                        logger.error(f"Error creating reserved subpart: {e}")
                else:
                    self.awaiting_subpart_metadata = True
                    self.subpart_metadata_start_line = line_num
                continue
            
            # === SECTION Detection ===
            section_match = re.match(
                r'^##\s*§?\s*(\d+\.\d+)\s*(?:[-–—:]\s+(.+))?$', 
                line, 
                re.IGNORECASE
            )
            if section_match:
                self._save_section_chunk()
                
                sec_num = section_match.group(1).strip()
                sec_title = section_match.group(2)
                
                is_reserved = "[Reserved]" in line or "[reserved]" in line.lower()
                
                if is_reserved or not sec_title or sec_title.strip() in ["", "[Reserved]"]:
                    # Reserved section
                    if self.current_part and self.current_subpart:
                        try:
                            chunk = ReservedSectionChunk(
                                type="reserved_section",
                                chunk_id=f"{sec_num}_reserved",
                                part=self.current_part,
                                subpart=self.current_subpart,
                                section=f"§ {sec_num}",
                                text="[Reserved]"
                            )
                            self.all_chunks.append(chunk)
                        except Exception as e:
                            logger.error(f"Error creating reserved section: {e}")
                else:
                    final_title = sec_title.strip()
                    
                    # Check for multi-line title
                    if final_title and not final_title.endswith('.'):
                        self.pending_section_header = (sec_num, final_title)
                        self.pending_section_line = line_num
                    else:
                        self._create_section_data(
                            sec_num, 
                            final_title or "No Title", 
                            line_num
                        )
                continue
            
            # === Collect metadata ===
            if self.awaiting_part_metadata:
                self.part_metadata_buffer.append(line)
                continue
            
            if self.awaiting_subpart_metadata:
                self.subpart_metadata_buffer.append(line)
                continue
            
            # === Section content ===
            if self.current_section_data is not None:
                self.section_text_buffer.append(line)
        
        # Finalize pending items
        if self.pending_section_header:
            sec_num, incomplete_title = self.pending_section_header
            self._create_section_data(sec_num, incomplete_title or "No Title", -1)
        
        self._save_section_chunk()
        self._save_part_metadata()
        self._save_subpart_metadata()
        
        return self.all_chunks
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics."""
        type_counts = {}
        subchunk_count = 0
        max_length = 0
        
        for chunk in self.all_chunks:
            chunk_type = chunk.type
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            
            if hasattr(chunk, 'text') and chunk.text:
                text_len = len(chunk.text)
                max_length = max(max_length, text_len)
            
            if hasattr(chunk, 'is_subchunk') and chunk.is_subchunk:
                subchunk_count += 1
        
        return {
            "total_chunks": len(self.all_chunks),
            "subchunk_count": subchunk_count,
            "max_length": max_length,
            "type_distribution": type_counts
        }


def parse_hipaa_file(filepath: str) -> List[ChunkType]:
    """Convenience function to parse a HIPAA text file."""
    parser = HIPAAParser()
    return parser.parse_file(filepath)


def parse_and_save(
    input_path: str,
    output_path: str,
    pretty: bool = True
) -> Dict[str, Any]:
    """
    Parse a file and save results to JSON.
    
    Args:
        input_path: Path to input text file
        output_path: Path for output JSON file
        pretty: Whether to pretty-print JSON
    
    Returns:
        Parsing statistics
    """
    parser = HIPAAParser()
    chunks = parser.parse_file(input_path)
    stats = parser.get_statistics()
    
    # Convert to JSON-serializable format
    json_output = [
        chunk.model_dump(exclude_none=True) 
        for chunk in chunks
    ]
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2 if pretty else None, ensure_ascii=False)
    
    logger.info(f"✅ Saved {len(chunks)} chunks to {output_path}")
    
    return stats


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python parser.py <input.txt> <output.json>")
        sys.exit(1)
    
    stats = parse_and_save(sys.argv[1], sys.argv[2])
    
    print(f"\n{'='*70}")
    print(f"Parsing Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Subchunks: {stats['subchunk_count']}")
    print(f"  Max length: {stats['max_length']} chars")
    print(f"\nType distribution:")
    for chunk_type, count in sorted(stats['type_distribution'].items()):
        print(f"  {chunk_type}: {count}")
    print(f"{'='*70}")
