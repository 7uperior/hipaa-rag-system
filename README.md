# HIPAA RAG System

Production-ready Retrieval-Augmented Generation system for querying HIPAA regulations.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Features

- **Hybrid Search**: Combines semantic vector search (60%) with keyword matching (40%)
- **AI Reranking**: LLM-based relevance scoring for optimal results
- **Async Architecture**: Non-blocking I/O with connection pooling
- **Citation Support**: Direct quotes with section references (§ XXX.XXX)
- **Vector Embeddings**: PostgreSQL + pgvector for semantic search
- **Professional UI**: Clean Gradio interface for easy interaction
- **Auto-Evaluation**: LLM-as-a-Judge testing framework (86.2% accuracy)

## 📊 System Architecture

```
User → Cloudflare Tunnel (HTTPS)
  ↓
NGINX Reverse Proxy :80
  ├── / → Gradio Frontend :7860
  └── /api/ → FastAPI Backend :8000
        ↓
  ├── PostgreSQL + pgvector :5432 (488 sections)
  └── OpenAI API (GPT-3.5-turbo + embeddings)
```

## 🚀 Quick Start

### Installation

1. **Clone and navigate:**
```bash
cd hipaa/
```

2. **Configure environment:**
```bash
cp .env.example .env
```
Add your OpenAI API key to .env
(Edit the file or run the command below)
```bash
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
```

3. **Start all services:**
```bash
docker compose up -d
```

4. **Load HIPAA data (first time only):**
```bash
# Parse PDF and generate embeddings (~1 minute, costs <$1)
docker compose exec backend python load_to_db.py

# Restart backend to activate
docker compose restart backend
```

5. **Verify system:**
```bash
curl http://localhost/api/
# Should return: {"status":"running","database":"postgresql_async","sections_loaded":488,"search_method":"hybrid_vector_keyword_reranking"}
```

### Access Points

- **Web UI**: http://localhost
- **API**: http://localhost/api/
- **API Docs**: http://localhost/api/docs

### Public Access (Optional)

```bash
# Using Cloudflare Tunnel
cloudflared tunnel --url http://localhost:80

# Or using ngrok
ngrok http 80
```

## 📖 Usage Examples

### Web Interface

1. Open http://localhost in browser
2. Enter question: "What is the purpose of HIPAA Part 160?"
3. Click "Ask Question"
4. View answer with cited sources

### API

```bash
# Ask a question
curl -X POST http://localhost/api/ask \
  -H "Content-Type: application/json" \
  -d '{"text":"What is the overall purpose of HIPAA Part 160?"}'

# Response
{
  "answer": "HIPAA Part 160 serves as the foundation for the Administrative Simplification provisions of the...",
  "sources": ["§ 160.101","§ 160.201","§ 164.534","§ 160.316","§ 164.500"]



### Python Client

```python
import requests

response = requests.post(
    "http://localhost/api/ask",
    json={"text": "What are civil penalties for noncompliance?"}
)

data = response.json()
print(data["answer"])
print("Sources:", data["sources"])
```

## 🧪 Testing



### Manual Testing

Test with these sample questions:
1.  What is the overall purpose of HIPAA Part 160?
2.	Which part covers data privacy measures?
3.	What does “minimum necessary” mean in HIPAA terminology?
4.	Which entities are specifically regulated under HIPAA?
5.	What are the potential civil penalties for noncompliance?
6.	Does HIPAA mention encryption best practices?
7.	Can I disclose personal health information to family members?
8.	If a covered entity outsources data processing, which sections apply?
9.	Cite the specific regulation texts regarding permitted disclosures to law enforcement.
10. Where security rule is located?
11. Give me full contents of 160.514 part 


## 🔧 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | Docker Compose | Container management |
| **Backend** | FastAPI + asyncio | Async REST API |
| **Database** | PostgreSQL 15 + pgvector | Vector storage & search |
| **Frontend** | Gradio | User interface |
| **Reverse Proxy** | NGINX | Request routing |
| **AI Model** | OpenAI GPT-3.5-turbo | Answer generation |
| **Embeddings** | text-embedding-3-small | Semantic search |
| **LLM Evaluation** | GPT-4o-mini | LLM-as-a-Judge grading |
| **Public Access** | Cloudflare Tunnel | HTTPS tunneling |


## 🛠️ Troubleshooting

### Backend shows 0 sections loaded

```bash
# Re-run PDF loading
docker compose exec backend python load_to_db.py
docker compose restart backend
```

### Frontend not accessible

```bash
# Check logs
docker compose logs frontend

# Restart frontend
docker compose restart frontend
```

### Database connection errors

```bash
# Restart all services
docker compose down
docker compose up -d
```

### OpenAI API errors

```bash
# Verify API key
docker compose exec backend printenv OPENAI_API_KEY

# Update .env and restart
docker compose restart backend
```

## 📝 Configuration

### Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-your-key-here
```

### Adjust Search Parameters

Edit `backend/main.py`:

```python
# Line ~270: Change top-k results
candidates = await hybrid_search(q.text, top_k=15)  # Default: 10

# Line ~271: Change reranking count
relevant = rerank_results(q.text, candidates, top_k=7)  # Default: 5

# Line ~272: Change context length
context_length = 2000  # Default: 1200
```

### Change AI Model

Edit `backend/main.py`:

```python
# Line ~310: Change to GPT-4
model="gpt-4"  # Default: gpt-3.5-turbo
```

# 🔮 Roadmap to >90% Accuracy
- **Refined Chunking Strategy**: Currently using section-based chunking. Optimization: Reduce chunk size to paragraph level. This reduces noise in the context window and improves retrieval precision.
- **Advanced Retrieval (GraphRAG)**. Implement GraphRAG (Knowledge Graph) to capture relationships between non-adjacent sections.
- **Query Expansion**: Generate multiple variations of the user's question to capture synonyms and legal terminology variations before searching the vector database.



## 🎓 Development

### Local Development

```bash
# View logs
docker compose logs -f backend

# Access container shell
docker compose exec backend bash

# Run tests
docker compose exec backend python test_evaluation.py

# Check database
docker compose exec postgres psql -U user -d hipaa
```

### Adding New Features

1. Edit source files in `backend/` or `frontend/`
2. Copy to container or rebuild:
```bash
# Quick update (no rebuild)
docker compose cp backend/main.py backend:/app/main.py
docker compose restart backend

# Or full rebuild
docker compose build backend
docker compose up -d
```

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- HIPAA regulations from HHS.gov
- OpenAI for GPT-3.5 and embeddings
- FastAPI framework
- Gradio UI library
- pgvector for PostgreSQL

## 📞 Support

For issues or questions:
1. Check logs: `docker compose logs`
2. Review troubleshooting section
3. Verify environment variables
4. Ensure OpenAI API credits available

## 🔄 Updates

**Version 1.0.0** (2025-11-28)
- Initial release
- Hybrid search implementation
- Async FastAPI backend
- LLM-based reranking
- 86.2% evaluation score


