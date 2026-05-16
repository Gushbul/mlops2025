# Multi-stage build for production optimization
FROM python:3.11-slim as base

# System dependencies and security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Poetry
RUN pip install poetry==1.8.3
RUN poetry config virtualenvs.create false

# Development stage
FROM base as development

WORKDIR /app

# Copy dependency files first (better Docker layer caching)
COPY pyproject.toml poetry.lock ./

# Install all dependencies including dev
RUN poetry install --no-interaction --no-ansi

# Copy source code
COPY src src
COPY configs configs
COPY tests tests
COPY Makefile ./

CMD ["python", "-m", "src.app.train"]

# Production stage
FROM base as production

# Install git for MLflow (optional but recommended)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy and install only production dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi --no-dev

# Copy source code and configs
COPY src src
COPY configs configs

# MLflow configuration
ENV MLFLOW_TRACKING_URI=file:./mlruns

# Create non-root user for security AFTER setting up files
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser

# Create mlruns directory and set permissions
RUN mkdir -p /app/mlruns && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src.app.train; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "src.app.train"]
