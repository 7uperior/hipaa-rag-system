FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir gradio requests

# Copy application
COPY . .

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
