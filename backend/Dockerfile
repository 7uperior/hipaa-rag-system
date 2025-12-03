FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    openai \
    PyMuPDF \
    pymupdf4llm \
    psycopg[binary] \
    asyncpg \
    pgvector \
    numpy \
    langchain \
    langchain-openai \
    langchain-community \
    langchain-neo4j \
    neo4j \
    requests \
    httpx
COPY . .
RUN mkdir -p /app/data

ENV PYTHONPATH=/app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]