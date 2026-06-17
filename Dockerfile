# ========== Stage 1: Builder ==========
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies only (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ========== Stage 2: Runtime ==========
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY chroma_db/ ./chroma_db/
COPY .env .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]