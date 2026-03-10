FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        fastapi \
        "uvicorn[standard]" \
        pydantic \
        chromadb \
        ollama \
        openai \
        trafilatura \
        beautifulsoup4 \
        lxml

EXPOSE 8000

CMD ["python", "api_main.py"]
