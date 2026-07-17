FROM python:3.10-slim

WORKDIR /workspace

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY backend/requirements.txt /workspace/backend/requirements.txt
RUN pip install --no-cache-dir -r /workspace/backend/requirements.txt

# Copy and install core tabnet package
COPY tabnet/ /workspace/tabnet/
RUN pip install --no-cache-dir -e /workspace/tabnet

# Copy backend application
COPY backend/ /workspace/backend/

# Ensure workspace folders exist
RUN mkdir -p /workspace/data/raw/uploaded \
    && mkdir -p /workspace/artifacts/checkpoints \
    && mkdir -p /workspace/artifacts/runs \
    && mkdir -p /workspace/artifacts/tensorboard

# Preload initial raw datasets and benchmarks
COPY data/ /workspace/data/
COPY benchmarks/ /workspace/benchmarks/

# Expose port and run uvicorn from /workspace context
EXPOSE 8000
WORKDIR /workspace
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
