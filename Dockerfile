# ========== Stage 1: Builder ==========
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ========== Stage 2: Runtime ==========
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY startup.sh .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

RUN chmod +x startup.sh

EXPOSE 7860

CMD ["./startup.sh"]