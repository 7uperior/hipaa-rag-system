"""
PDF Extractor
=============
Extracts and linearizes text from 3-column HIPAA PDF documents.
Handles table of contents detection, column extraction, and artifact cleaning.
"""

import re
from pathlib import Path
from typing import Optional

import pdfplumber

from config import get_etl_logger

logger = get_etl_logger()


class PDFExtractor:
    """
    Extracts text from multi-column PDF documents.
    
    Optimized for HIPAA regulation documents which typically use
    a 3-column layout.
    """
    
    def __init__(
        self,
        top_margin: int = 60,
        bottom_margin: int = 50,
        column_padding: int = 2,
        x_tolerance: int = 1,
        y_tolerance: int = 3
    ):
        """
        Initialize the PDF extractor.
        
        Args:
            top_margin: Pixels to skip from top of page
            bottom_margin: Pixels to skip from bottom of page
            column_padding: Padding between columns
            x_tolerance: Character grouping tolerance (horizontal)
            y_tolerance: Line grouping tolerance (vertical)
        """
        self.top_margin = top_margin
        self.bottom_margin = bottom_margin
        self.column_padding = column_padding
        self.x_tolerance = x_tolerance
        self.y_tolerance = y_tolerance
    
    def is_toc_page(self, text: str) -> bool:
        """
        Determine if a page is a table of contents.
        
        Args:
            text: Raw page text
        
        Returns:
            True if page appears to be TOC
        """
        if not text:
            return False
        return text.count('....') > 10 or "Contents" in text[:200]
    
    def clean_artifacts(self, text: str) -> str:
        """
        Remove common artifacts from extracted text.
        
        Args:
            text: Raw extracted text
        
        Returns:
            Cleaned text
        """
        # Remove header artifacts
        text = text.replace(
            "HIPAA Administrative Simplification Regulation Text March 2013", 
            ""
        )
        text = text.replace(
            "HIPAA Administrative Simplification Regulation Text\nMarch 2013", 
            ""
        )
        
        # Remove standalone page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        return text
    
    def extract_page_columns(self, page, num_columns: int = 3) -> str:
        """
        Extract text from a multi-column page.
        
        Args:
            page: pdfplumber page object
            num_columns: Number of columns to extract
        
        Returns:
            Combined text from all columns
        """
        width = page.width
        height = page.height
        col_width = width / num_columns
        
        page_text = []
        
        for i in range(num_columns):
            left = i * col_width + self.column_padding
            right = (i + 1) * col_width - self.column_padding
            
            bbox = (left, self.top_margin, right, height - self.bottom_margin)
            
            try:
                col_crop = page.crop(bbox)
                text = col_crop.extract_text(
                    x_tolerance=self.x_tolerance,
                    y_tolerance=self.y_tolerance
                )
                if text:
                    page_text.append(text)
            except ValueError as e:
                logger.warning(f"Error extracting column {i}: {e}")
        
        return "\n\n".join(page_text)
    
    def extract(
        self,
        pdf_path: str,
        skip_first_page: bool = True,
        num_columns: int = 3
    ) -> str:
        """
        Extract and linearize text from a multi-column PDF.
        
        Args:
            pdf_path: Path to the PDF file
            skip_first_page: Whether to skip the title page
            num_columns: Number of columns per page
        
        Returns:
            Linearized text content
        """
        logger.info(f"📖 Processing PDF: {pdf_path}")
        
        full_text = ""
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                # Skip title page
                if skip_first_page and i == 0:
                    logger.debug("Skipping title page")
                    continue
                
                # Progress logging
                if (i + 1) % 20 == 0:
                    logger.info(f"   Processing page {i + 1}/{total_pages}")
                
                # Check for TOC
                raw_text = page.extract_text() or ""
                if self.is_toc_page(raw_text):
                    logger.debug(f"Skipping TOC page {i + 1}")
                    continue
                
                # Extract columns
                page_content = self.extract_page_columns(page, num_columns)
                
                # Clean artifacts
                page_content = self.clean_artifacts(page_content)
                
                full_text += page_content + "\n"
        
        # Final cleanup
        full_text = full_text.replace('\xa0', ' ')
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        
        logger.info(f"✅ Extracted {len(full_text)} characters")
        
        return full_text
    
    def extract_to_file(
        self,
        pdf_path: str,
        output_path: str,
        **kwargs
    ) -> Path:
        """
        Extract PDF text and save to file.
        
        Args:
            pdf_path: Path to input PDF
            output_path: Path for output text file
            **kwargs: Additional arguments for extract()
        
        Returns:
            Path to output file
        """
        text = self.extract(pdf_path, **kwargs)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logger.info(f"💾 Saved to: {output_file.absolute()}")
        
        return output_file


def process_pdf(
    pdf_path: str,
    output_path: str,
    **kwargs
) -> Path:
    """
    Convenience function to process a PDF file.
    
    Args:
        pdf_path: Path to input PDF
        output_path: Path for output text file
        **kwargs: Additional arguments for PDFExtractor
    
    Returns:
        Path to output file
    """
    extractor = PDFExtractor(**kwargs)
    return extractor.extract_to_file(pdf_path, output_path)


if __name__ == "__main__":
    # CLI usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python pdf.py <input.pdf> <output.txt>")
        sys.exit(1)
    
    process_pdf(sys.argv[1], sys.argv[2])
