FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    openai \
    asyncpg \
    pgvector \
    pydantic \
    pydantic-settings \
    httpx

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Run startup script (loads data + starts server)
CMD ["python", "backend/startup.py"]
