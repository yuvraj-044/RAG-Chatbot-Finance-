# Dockerfile
# -----------
# Used by Hugging Face Spaces to build and run this FastAPI app.
# HF Spaces expects the app to listen on port 7860 — do NOT change that.
#
# Build locally to test: docker build -t finance-rag-api .
# Run locally:           docker run -p 7860:7860 --env-file .env finance-rag-api

# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed by ChromaDB (it has a C extension)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (Docker layer caching — this
# layer only rebuilds when requirements.txt changes, not on every code change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create the chroma_db directory so ChromaDB can persist data
# (if your teammate provides a pre-built index, it should live here)
RUN mkdir -p chroma_db

# HF Spaces requires port 7860
EXPOSE 7860

# Start the FastAPI server via uvicorn
# --host 0.0.0.0  → listen on all interfaces (required inside Docker)
# --port 7860     → HF Spaces expects this exact port
# --workers 1     → single worker is fine for a hackathon demo
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
