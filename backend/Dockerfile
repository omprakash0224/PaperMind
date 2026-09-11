# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install deps into a clean prefix so we can copy only the wheels.
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools needed for some wheels (e.g. tiktoken Rust extension)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

# Runtime directories (overridden by volume mounts in compose)
RUN mkdir -p uploads qdrant_data

# Non-root user for security
RUN useradd --no-create-home --shell /bin/false appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Default: run FastAPI.
# Override in docker-compose for the ARQ worker:
#   command: arq app.worker.WorkerSettings
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
