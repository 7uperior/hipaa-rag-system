"""Text transformation and cleaning."""

import re


class TextCleaner:
    """Clean PDF artifacts from extracted text."""
    
    def clean(self, raw_text: str) -> str:
        """Clean extracted text from PDF artifacts.
        
        Removes:
        - Page headers/footers
        - Page numbers
        - Table of contents sections
        - Fixes line breaks
        
        Args:
            raw_text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        print("🧹 Cleaning text...")
        
        lines = raw_text.split('\n')
        cleaned_lines = []
        
        in_contents = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                in_contents = False
                continue
            
            # Filter page artifacts
            if self._is_artifact(line):
                continue
            
            # Handle TOC
            if self._is_toc_marker(line):
                in_contents = True
                continue
            
            if in_contents:
                if line.startswith('§ ') and not line.endswith('.'):
                    in_contents = False
                else:
                    continue
            
            # Fix line breaks
            if cleaned_lines and self._should_merge(cleaned_lines[-1], line):
                cleaned_lines[-1] += ' ' + line
                continue
            
            cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        print(f"✅ Cleaned to {len(cleaned_text):,} characters\n")
        
        return cleaned_text
    
    def _is_artifact(self, line: str) -> bool:
        """Check if line is a PDF artifact."""
        # Page headers
        if 'HIPAA Administrative Simplification' in line:
            return True
        if line in ('March 2013', 'March 2013 '):
            return True
        
        # Standalone page numbers
        if re.match(r'^\d{1,3}$', line):
            return True
        
        # TOC entries
        if re.match(r'^§\s*\d+\.\d+\s+[\w\s]{1,60}\.$', line):
            return True
        
        # Subpart headers
        if line.startswith('Subpart ') and len(line.split()) < 10:
            return True
        
        # Appendix references
        if line.startswith('Appendix ') and 'to' in line:
            return True
        
        return False
    
    def _is_toc_marker(self, line: str) -> bool:
        """Check if line marks TOC section."""
        if line in ('Contents', 'CONTENTS'):
            return True
        if line.startswith('PART ') and '—' in line:
            return True
        return False
    
    def _should_merge(self, prev_line: str, curr_line: str) -> bool:
        """Check if current line should merge with previous."""
        # Don't merge if previous line ended with punctuation
        if prev_line.endswith(('.', ':', ';', ')', '?', '!')):
            return False
        
        # Don't merge if current line starts with section marker
        if curr_line.startswith(('(', '§')):
            return False
        
        return True


class SectionParser:
    """Parse text into hierarchical sections."""
    
    def __init__(self):
        """Initialize parser."""
        self.main_pattern = re.compile(r'^§\s*(\d+\.\d+)\s+(.+)')
        self.level2_pattern = re.compile(r'^\(([a-z])\)\s+(.+)')
        self.level3_pattern = re.compile(r'^\((\d+)\)\s+(.+)')
        self.level4_pattern = re.compile(r'^\(([ivxlc]+)\)\s+(.+)')
    
    def parse(self, text: str) -> list[dict]:
        """Parse text into hierarchical chunks.
        
        Args:
            text: Cleaned text
            
        Returns:
            List of section chunks
        """
        print("🔪 Splitting into hierarchical chunks...")
        
        chunks_dict = {}
        lines = text.split('\n')
        
        # State tracking
        current_main_section: str | None = None
        current_level2_section: str | None = None
        current_level3_section: str | None = None
        current_level4_section: str | None = None
        current_content: list[str] = []
        
        def save_chunk():
            """Save current chunk to dict."""
            if not current_content:
                return
            
            active_section = (
                current_level4_section or
                current_level3_section or 
                current_level2_section or 
                current_main_section
            )
            
            if not active_section:
                return
            
            content = ' '.join(current_content).strip()
            
            # Skip too short or noise
            if len(content) < 30 or content.startswith('...'):
                return
            
            word_count = len(content.split())
            
            chunks_dict[active_section] = {
                'section': active_section,
                'parent': current_main_section or active_section,
                'content': content,
                'word_count': word_count,
                'level': active_section.count('(') + 1
            }
        
        for line in lines:
            line = line.strip()
            
            if not line or len(line) < 3:
                continue
            
            # Check for main section
            main_match = self.main_pattern.match(line)
            if main_match:
                save_chunk()
                current_main_section = main_match.group(1)
                current_level2_section = None
                current_level3_section = None
                current_level4_section = None
                current_content = []
                continue
            
            if not current_main_section:
                continue
            
            # Check for Level 2
            level2_match = self.level2_pattern.match(line)
            if level2_match:
                save_chunk()
                sub_id = level2_match.group(1)
                current_level2_section = f"{current_main_section}({sub_id})"
                current_level3_section = None
                current_level4_section = None
                current_content = []
                continue
            
            # Check for Level 3
            level3_match = self.level3_pattern.match(line)
            if level3_match and current_level2_section:
                save_chunk()
                sub_id = level3_match.group(1)
                current_level3_section = f"{current_level2_section}({sub_id})"
                current_level4_section = None
                current_content = []
                continue
            
            # Check for Level 4
            level4_match = self.level4_pattern.match(line)
            if level4_match and current_level3_section:
                save_chunk()
                sub_id = level4_match.group(1)
                current_level4_section = f"{current_level3_section}({sub_id})"
                current_content = []
                continue
            
            current_content.append(line)
        
        save_chunk()
        
        chunks = list(chunks_dict.values())
        print(f"✅ Created {len(chunks)} unique chunks\n")
        
        return chunks
    
    def analyze(self, chunks: list[dict]) -> None:
        """Print chunk statistics."""
        if not chunks:
            return
        
        print("=" * 70)
        print("📊 CHUNK ANALYSIS")
        print("=" * 70)
        
        level_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        for chunk in chunks:
            level = chunk.get('level', 1)
            level_counts[level] += 1
        
        print(f"\n📈 Total chunks: {len(chunks)}")
        print("\n📊 Distribution by level:")
        print(f"   Level 1 (§ X.Y):           {level_counts[1]:3d} chunks")
        print(f"   Level 2 (§ X.Y(a)):        {level_counts[2]:3d} chunks")
        print(f"   Level 3 (§ X.Y(a)(1)):     {level_counts[3]:3d} chunks")
        print(f"   Level 4 (§ X.Y(a)(1)(i)):  {level_counts[4]:3d} chunks")
        
        word_counts = [c['word_count'] for c in chunks]
        avg = sum(word_counts) / len(word_counts)
        
        print("\n📝 Content statistics:")
        print(f"   Average words: {avg:.0f}")
        print(f"   Min words: {min(word_counts)}")
        print(f"   Max words: {max(word_counts)}")
        
        # Find § 164.512
        law_chunks = [c for c in chunks if '164.512' in c['section']]
        if law_chunks:
            print("\n🎯 Found § 164.512 sub-sections:")
            for chunk in law_chunks[:10]:
                print(f"   § {chunk['section']:25s} {chunk['word_count']:4d} words")
        
        print("=" * 70 + "\n")