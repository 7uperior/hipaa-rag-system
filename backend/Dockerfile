FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    openai \
    pypdf2 \
    psycopg[binary] \
    asyncpg \
    pgvector \
    numpy \
    langchain \
    langchain-openai \
    requests \
    httpx

COPY . .
RUN mkdir -p /app/data

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]