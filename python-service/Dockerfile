FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Health check using Railway's PORT env var
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import os, requests; requests.get(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/health')"

# Run FastAPI with uvicorn using Railway's PORT env var
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

