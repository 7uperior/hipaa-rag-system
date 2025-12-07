#!/usr/bin/env python
"""
HIPAA ETL Pipeline Runner
=========================

Orchestrates the full ETL process:
1. Extract: Linearize PDF to text
2. Transform: Parse text into structured chunks
3. Load: Insert chunks with embeddings into PostgreSQL

Usage:
    python run_etl.py <input.pdf> [--skip-pdf] [--text-file <path>]
    
Examples:
    # Full pipeline from PDF
    python run_etl.py data/hipaa_regulations.pdf
    
    # Skip PDF extraction (use existing text file)
    python run_etl.py data/hipaa_regulations.pdf --skip-pdf --text-file data/linearized.txt
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import setup_logging, get_settings
from ETL.extractors import PDFExtractor
from ETL.transformers import HIPAAParser
from ETL.loaders import PostgresLoader

logger = setup_logging(name="hipaa.etl")
settings = get_settings()


async def run_pipeline(
    pdf_path: str,
    skip_pdf: bool = False,
    text_file: str = None,
    output_dir: str = "data"
):
    """
    Run the complete ETL pipeline.
    
    Args:
        pdf_path: Path to input PDF file
        skip_pdf: Whether to skip PDF extraction
        text_file: Path to pre-extracted text file (if skip_pdf)
        output_dir: Directory for intermediate files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Derive file paths
    pdf_name = Path(pdf_path).stem
    text_path = text_file or str(output_path / f"{pdf_name}_linearized.txt")
    json_path = str(output_path / f"{pdf_name}_chunks.json")
    
    print("="*70)
    print("🏥 HIPAA ETL Pipeline")
    print("="*70)
    
    # Step 1: Extract PDF
    if not skip_pdf:
        print("\n📖 Step 1: Extracting PDF...")
        extractor = PDFExtractor()
        extractor.extract_to_file(pdf_path, text_path)
        print(f"   ✅ Saved to: {text_path}")
    else:
        print(f"\n📖 Step 1: Skipping PDF extraction (using: {text_path})")
    
    # Step 2: Parse text into chunks
    print("\n🔧 Step 2: Parsing into chunks...")
    parser = HIPAAParser()
    chunks = parser.parse_file(text_path)
    stats = parser.get_statistics()
    
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Subchunks: {stats['subchunk_count']}")
    print(f"   Max length: {stats['max_length']} chars")
    print(f"   Types: {stats['type_distribution']}")
    
    # Save to JSON (intermediate step)
    import json
    json_output = [
        chunk.model_dump(exclude_none=True)
        for chunk in chunks
    ]
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Saved chunks to: {json_path}")
    
    # Step 3: Load into database
    print("\n📤 Step 3: Loading into PostgreSQL...")
    loader = PostgresLoader()
    
    try:
        await loader.connect()
        
        # Create schema
        print("   Creating schema...")
        await loader.create_schema()
        await loader.create_references_table()
        
        # Load chunks
        print("   Loading chunks with embeddings...")
        load_stats = await loader.load_chunks(json_output, batch_size=5)
        
        # Create indexes and functions
        print("   Creating indexes...")
        await loader.create_vector_index()
        await loader.create_helper_functions()
        await loader.populate_references()
        
        # Final count
        count = await loader.conn.fetchval("SELECT COUNT(*) FROM hipaa_sections")
        
        print(f"\n   ✅ Successfully loaded!")
        print(f"   Successful: {load_stats['successful']}")
        print(f"   Failed: {load_stats['failed']}")
        print(f"   Total in DB: {count}")
    
    finally:
        await loader.close()
    
    print("\n" + "="*70)
    print("✅ ETL Pipeline Complete!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - Linearized text: {text_path}")
    print(f"  - JSON chunks: {json_path}")
    print(f"  - Database: PostgreSQL (hipaa_sections table)")
    print("\nYou can now start the API server with:")
    print("  docker-compose up")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HIPAA ETL Pipeline - Process regulations PDF into searchable database"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to HIPAA regulations PDF"
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip PDF extraction (use existing text file)"
    )
    parser.add_argument(
        "--text-file",
        help="Path to pre-extracted text file (requires --skip-pdf)"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for output files (default: data)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.text_file and not args.skip_pdf:
        parser.error("--text-file requires --skip-pdf")
    
    if args.skip_pdf and not args.text_file:
        # Default to derived path
        pdf_name = Path(args.pdf_path).stem
        args.text_file = f"{args.output_dir}/{pdf_name}_linearized.txt"
    
    # Run pipeline
    asyncio.run(run_pipeline(
        pdf_path=args.pdf_path,
        skip_pdf=args.skip_pdf,
        text_file=args.text_file,
        output_dir=args.output_dir
    ))


if __name__ == "__main__":
    main()
