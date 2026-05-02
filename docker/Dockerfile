# MLOps Portfolio Container Image
# Python 3.12 slim base with uv package manager
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency files and source code
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Install Python dependencies into system Python (not venv)
# This ensures 'python -m ...' commands work without 'uv run' prefix
RUN uv pip install --system --no-cache .

# Copy pipeline configuration
COPY dvc.yaml params.yaml ./
COPY k8s/ k8s/
COPY great_expectations/gx/ great_expectations/gx/

# Set Python path for module imports
ENV PYTHONPATH=/app

# Note: mlruns/, data/, and models/ directories are NOT copied
# These will be mounted from the Kubernetes PVC at runtime

# Default command: show help
CMD ["python", "--version"]
