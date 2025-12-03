"""PDF text extraction."""

import fitz


class PDFExtractor:
    """Extract text from PDF files."""
    
    def __init__(self, pdf_path: str, skip_first_pages: int = 9):
        """Initialize extractor.
        
        Args:
            pdf_path: Path to PDF file
            skip_first_pages: Number of pages to skip (TOC)
        """
        self.pdf_path = pdf_path
        self.skip_first_pages = skip_first_pages
    
    def extract(self) -> str:
        """Extract text from PDF.
        
        Returns:
            Raw text from PDF
        """
        print(f"📖 Reading PDF: {self.pdf_path}")
        
        doc = fitz.open(self.pdf_path)
        total_pages = len(doc)
        start_page = self.skip_first_pages
        
        print(f"   Total pages: {total_pages}")
        print(f"   Skipping: pages 1-{start_page} (TOC)")
        print(f"   Processing: pages {start_page + 1}-{total_pages}")
        
        # Extract raw text
        raw_text_parts = []
        for page_num in range(start_page, total_pages):
            page = doc[page_num]
            text = page.get_text("text")
            raw_text_parts.append(text)
            
            if (page_num - start_page + 1) % 50 == 0:
                processed = page_num - start_page + 1
                remaining = total_pages - start_page
                print(f"   Progress: {processed}/{remaining} pages...")
        
        doc.close()
        
        raw_text = '\n'.join(raw_text_parts)
        print(f"✅ Extracted {len(raw_text):,} characters")
        
        return raw_text