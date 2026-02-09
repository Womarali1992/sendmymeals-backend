# Single-stage build for App Platform compatibility
FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than GPU version)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model_loader.py .
COPY multi_model_server.py .

# Pre-download models during build (optional, makes container larger but faster startup)
# RUN python -c "from transformers import pipeline; pipeline('image-classification', model='nateraw/food')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the server (no reload in production)
CMD ["python", "-c", "import uvicorn; uvicorn.run('multi_model_server:app', host='0.0.0.0', port=8000, workers=1)"]
