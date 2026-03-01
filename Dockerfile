FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv \
    ffmpeg libsndfile1 \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" || pip install --no-cache-dir .

# Copy application code
COPY . .
RUN pip install --no-cache-dir -e .

# Create data directories
RUN mkdir -p /app/data/jobs /app/data/tmp

EXPOSE 8000

# Default: run API server
CMD ["uvicorn", "vaanidub.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
