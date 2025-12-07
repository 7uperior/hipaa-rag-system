# 🏥 HIPAA RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for querying HIPAA regulations (Parts 160, 162, and 164).

## Features

- **Hybrid Search**: Combines semantic vector search (60%) with keyword matching (40%)
- **LLM Reranking**: GPT-4o-mini-based relevance scoring for improved accuracy
- **Query Classification**: Routes queries to specialized handlers (full_text, citation, explanation, reference_list)
- **Intelligent Chunking**: Hierarchical chunking that respects legal document structure
- **Citation Support**: Proper § citation formatting with section references
- **Async Architecture**: High-performance async database operations with connection pooling
- **Auto Data Loading**: Automatically loads pre-processed data on first startup

## Architecture

```
hipaa/
├── config/                     # Configuration management
│   ├── settings.py            # Pydantic settings (centralized config)
│   └── logging.py             # Logging configuration
│
├── ETL/                        # Extract-Transform-Load pipeline
│   ├── models/
│   │   └── chunks.py          # Pydantic models for chunk types
│   ├── extractors/
│   │   └── pdf.py             # 3-column PDF linearization
│   ├── transformers/
│   │   ├── parser.py          # Text → structured chunks
│   │   └── chunker.py         # Intelligent section splitting
│   └── loaders/
│       └── postgres.py        # PostgreSQL + pgvector loading
│
├── backend/                    # FastAPI API server
│   ├── main.py                # Routes and application setup
│   ├── startup.py             # Auto-load data + start server
│   ├── dependencies.py        # FastAPI dependency injection
│   ├── db/
│   │   ├── connection.py      # Async connection pool
│   │   └── queries.py         # SQL query definitions
│   ├── services/
│   │   ├── search.py          # Vector/keyword/hybrid search
│   │   ├── reranker.py        # LLM-based reranking
│   │   ├── classifier.py      # Query type classification
│   │   └── generator.py       # Answer generation
│   └── models/
│       └── schemas.py         # API request/response models
│
├── frontend/                   # Gradio web interface
│   └── app.py                 # Chat interface
│
├── nginx/                      # Reverse proxy
│   └── nginx.conf
│
├── tests/                      # Test suite
│   ├── test_chunking.py
│   ├── test_search.py
│   └── evaluation/
│       └── test_evaluation.py  # LLM-as-Judge evaluation
│
├── data/                       # Pre-processed data
│   └── hipaa_data.json        # Chunks ready for loading
│
├── docker-compose.yml          # Container orchestration
├── run_etl.py                  # ETL pipeline runner
└── requirements.txt
```

## Quick Start

### Prerequisites

- Docker & Docker Compose
- OpenAI API key

### 1. Setup Environment

```bash
cd hipaa

# Create environment file
cp .env.example .env

# Add your OpenAI API key to .env
# OPENAI_API_KEY=sk-your-key-here
```

### 2. Start Services

```bash
# Start all services (data loads automatically on first run)
docker-compose up -d

# View logs to monitor data loading
docker-compose logs -f backend
```

The backend will automatically:
1. Wait for PostgreSQL to be ready
2. Check if data is already loaded
3. Load `data/hipaa_data.json` if database is empty
4. Create vector indexes
5. Start the API server

### 3. Access the System

- **Web UI**: http://localhost (via NGINX)
- **API**: http://localhost:8000
- **Gradio Direct**: http://localhost:7860

### 4. Public Access (Optional)

To expose the system to the internet via Cloudflare Tunnel:

```bash
# Start with cloudflared tunnel
docker-compose --profile tunnel up -d

# Watch logs to get the public URL
docker-compose logs -f cloudflared
```

You'll see a URL like `https://random-name.trycloudflare.com` that provides secure public access.

## API Endpoints

### Ask Question

```bash
POST /ask
Content-Type: application/json

{
    "text": "What does minimum necessary mean?"
}
```

Response:
```json
{
    "answer": "Under HIPAA, 'minimum necessary' refers to...",
    "sources": ["§ 164.502", "§ 164.514"],
    "query_type": "explanation"
}
```

### Search Sections

```bash
GET /search/{query}
```

### Get Full Section

```bash
GET /section/160.514
```

### Health Check

```bash
GET /
```

## Query Types

The system classifies queries into four types:

| Type | Description | Example |
|------|-------------|---------|
| `full_text` | Complete section retrieval | "Give me full contents of 164.530" |
| `citation` | Exact quotes with references | "Quote the text about disclosures" |
| `explanation` | Plain language explanation | "What does minimum necessary mean?" |
| `reference_list` | List of relevant sections | "Which sections cover security?" |

## Configuration

All settings can be configured via environment variables:

### Search Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_VECTOR_WEIGHT` | 0.6 | Weight for semantic search |
| `SEARCH_KEYWORD_WEIGHT` | 0.4 | Weight for keyword search |
| `SEARCH_TOP_K` | 15 | Initial candidates to retrieve |

### Database Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | postgres | Database host |
| `DB_PORT` | 5432 | Database port |
| `DB_NAME` | hipaa | Database name |
| `DB_USER` | user | Database user |
| `DB_PASSWORD` | pass | Database password |

### Model Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `EMBEDDING_MODEL` | text-embedding-3-small | OpenAI embedding model |
| `RERANK_MODEL` | gpt-4o-mini | Model for reranking |
| `GENERATION_MODEL` | gpt-3.5-turbo | Model for answers |

## Local Development

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Start database only
docker-compose up postgres -d

# Run backend locally
uv run uvicorn backend.main:app --reload

# Run frontend locally  
uv run python frontend/app.py
```

## Evaluation

Run the LLM-as-Judge evaluation suite:

```bash
# Ensure backend is running
docker-compose up -d

# Run evaluation
uv run python tests/evaluation/test_evaluation.py
```

Metrics evaluated:
- Accuracy
- Completeness
- Citation quality
- Relevance
- Query type fit

## Run Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_chunking.py -v
```

## Performance

Current benchmarks (12 test questions):
- **Overall Accuracy**: 86.2%
- **Explanation queries**: 4.3/5
- **Citation queries**: 4.1/5
- **Reference list queries**: 4.4/5

---

## Advanced: ETL Pipeline (Optional)

If you need to re-process the HIPAA PDF from scratch:

### Prerequisites
- HIPAA regulations PDF (download from HHS.gov)

### Run ETL Pipeline

```bash
# Install dependencies
uv sync

# Run full pipeline from PDF
uv run python run_etl.py data/hipaa_regulations.pdf

# Or with existing text file
uv run python run_etl.py data/hipaa.pdf --skip-pdf --text-file data/hipaa_linear_text.txt
```

### Chunking Strategy

The system uses intelligent hierarchical chunking:

1. **Part Level**: Captures Part metadata (authority, source)
2. **Subpart Level**: Groups related sections
3. **Section Level**: Primary content unit
4. **Subsection Grouping**: Groups (a), (b), (c) into ~5-7k char chunks

```
§ 164.530 Administrative Requirements
├── chunk_1: (a)-(c) [5,200 chars]
├── chunk_2: (d)-(f) [4,800 chars]
└── chunk_3: (g)-(i) [6,100 chars]
```

### Database Schema

```sql
CREATE TABLE hipaa_sections (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(100) UNIQUE NOT NULL,
    chunk_type VARCHAR(50) NOT NULL,
    
    -- Hierarchy
    part VARCHAR(10) NOT NULL,
    subpart VARCHAR(50),
    section VARCHAR(50),
    section_title TEXT,
    
    -- Subchunk metadata
    is_subchunk BOOLEAN DEFAULT FALSE,
    parent_section VARCHAR(100),
    grouped_subsections TEXT[],
    
    -- Content
    text TEXT NOT NULL,
    cross_references TEXT[],
    
    -- Vector
    embedding vector(1536)
);
```

---

## Roadmap

- [ ] **Paragraph-level chunking**: Reduce noise in results
- [ ] **GraphRAG**: Utilize cross-reference relationships
- [ ] **Query expansion**: Legal terminology variations
- [ ] **Citation validation**: Verify cited sections exist
- [ ] **Multi-document support**: Support for related regulations

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- HIPAA regulation text from [HHS.gov](https://www.hhs.gov/hipaa/for-professionals/privacy/laws-regulations/combined-regulation-text/index.html)
- Built with FastAPI, Gradio, PostgreSQL + pgvector, and OpenAI
