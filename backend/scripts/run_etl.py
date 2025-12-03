"""Main ETL pipeline."""

import asyncio
from etl.extractors import PDFExtractor
from etl.transformers import TextCleaner, SectionParser
from etl.loaders import JSONLoader, PostgreSQLLoader


async def run_etl(
    pdf_path: str = '/app/data/hipaa_combined.pdf',
    json_path: str = '/app/data/hipaa_data.json',
    skip_first_pages: int = 9
) -> None:
    """Run full ETL pipeline.
    
    Args:
        pdf_path: Path to input PDF
        json_path: Path to output JSON
        skip_first_pages: Pages to skip (TOC)
    """
    print("\n" + "=" * 70)
    print("🚀 HIPAA ETL PIPELINE")
    print("=" * 70 + "\n")
    
    # Extract
    extractor = PDFExtractor(pdf_path, skip_first_pages)
    raw_text = extractor.extract()
    
    # Transform
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean(raw_text)
    
    parser = SectionParser()
    chunks = parser.parse(cleaned_text)
    parser.analyze(chunks)
    
    # Load to JSON
    json_loader = JSONLoader()
    json_loader.save(chunks, json_path)
    
    # Load to PostgreSQL
    pg_loader = PostgreSQLLoader()
    await pg_loader.connect()
    await pg_loader.setup_database()
    await pg_loader.load(chunks)
    await pg_loader.close()
    
    print("\n" + "=" * 70)
    print("✅ ETL COMPLETE")
    print("=" * 70)
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"JSON file: {json_path}")
    print(f"PostgreSQL: sections table\n")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(run_etl())