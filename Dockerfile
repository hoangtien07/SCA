# ─── FastAPI Backend ──────────────────────────────────────────────────────────
# Build: docker build -t sca-api .
# Run:   docker run --env-file .env -p 8000:8000 sca-api
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for chromadb (sqlite3, build tools)
RUN apt-get update && \
  apt-get install -y --no-install-recommends gcc g++ && \
  rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config/ config/
COPY src/ src/
COPY data/ data/
COPY .env* ./

# Ensure data directories exist
RUN mkdir -p data/raw data/processed data/knowledge_base

EXPOSE 8000

# Railway injects PORT env var; default to 8000
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
