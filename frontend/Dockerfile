FROM python:3.11-slim

WORKDIR /app

RUN pip install gradio requests

COPY . .

CMD ["python", "app.py"]